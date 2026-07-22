# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA, symmetric_kl_laplace
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import autocast

from .metrics import bbox_iou, probiou, WiseIouLoss
from .tal import bbox2dist
import os


class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss

class VarifocalLoss_YOLO(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self, alpha=0.75, gamma=2.0):
        """Initialize the VarifocalLoss class."""
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_score, gt_score):
        """Computes varfocal loss."""
        
        weight = self.alpha * (pred_score.sigmoid() - gt_score).abs().pow(self.gamma) * (gt_score <= 0.0).float() + gt_score * (gt_score > 0.0).float()
        with torch.cuda.amp.autocast(enabled=False):
            return F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction='none') * weight

class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class DFLoss(nn.Module):
    """Criterion class for computing DFL losses during training."""

    def __init__(self, reg_max=16) -> None:
        """Initialize the DFL module."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)

def gbb_form_hbb(xywh):
    """
    为水平边界框 (x,y,w,h) 转换为高斯参数形式
    返回: [x, y, a, b, c]  where a = w²/12, b = h²/12, c = 0 (无旋转)
    """
    x, y, w, h = xywh.unbind(-1)
    a = (w ** 2) / 12.0
    b = (h ** 2) / 12.0
    c = torch.zeros_like(w)  # 无旋转，协方差 off-diagonal = 0
    return torch.stack([x, y, a, b, c], dim=-1)


def rotated_form_hbb(a_, b_, c_):
    """
    HBB 场景下直接返回原值（因为没有旋转）
    """
    return a_, b_, c_


def probiou_loss(pred, target, eps=1e-4, mode='l1',gamma=0.5):
    """
    ProbIoU 损失 - 专为水平边界框 (HBB) 优化版本
    pred, target: [N, 4] 或 [N, 5]，格式为 (x, y, w, h) 或 (x, y, w, h, angle)
                 如果是 [N,4]，自动补 angle=0
    mode: 'l1' → 范围约 [0,1)，后20%代使用
          'l2' → 范围 [0, +inf)，前80%代使用
    """
    if pred.shape[-1] == 4:
        pred = torch.cat([pred, torch.zeros_like(pred[..., :1])], dim=-1)  #补 angle=0
    if target.shape[-1] == 4:
        target = torch.cat([target, torch.zeros_like(target[..., :1])], dim=-1)

    # 转换为高斯参数
    g1 = gbb_form_hbb(pred[..., :4])
    g2 = gbb_form_hbb(target[..., :4])

    x1, y1, a1, b1, c1 = g1.unbind(-1)
    x2, y2, a2, b2, c2 = g2.unbind(-1)

    # HBB 场景 rotated_form 其实没必要，但保留接口一致性
    a1, b1, c1 = rotated_form_hbb(a1, b1, c1)
    a2, b2, c2 = rotated_form_hbb(a2, b2, c2)

    # Bhattacharyya 距离的近似计算
    denom = (a1 + a2) * (b1 + b2) - (c1 + c2) ** 2 + eps

    t1 = (
        (a1 + a2) * (y1 - y2) ** 2 +
        (b1 + b2) * (x1 - x2) ** 2
    ) / denom * 0.25

    t2 = (
        (c1 + c2) * (x2 - x1) * (y1 - y2)
    ) / denom * 0.5

    det_prod = torch.sqrt((a1 * b1 - c1 ** 2) * (a2 * b2 - c2 ** 2) + eps)
    t3 = torch.log(
        ((a1 + a2) * (b1 + b2) - (c1 + c2) ** 2) /
        (4 * det_prod + eps) + eps
    ) * 0.5

    B_d = t1 + t2 + t3
    B_d = torch.clamp(B_d, min=0.0, max=50.0)   # 更合理的上限，避免极端值

    bc = torch.exp(-B_d)
    #原版probiou
    # l1 = torch.sqrt(1.0 - bc + eps)
    
    # if mode == 'l1':
    #     loss = l1
    # elif mode == 'l2':
    #     li = l1 ** 2
    #     loss = -torch.log(1.0 - li + eps)
    # else:
    #     raise ValueError("mode must be 'l1' or 'l2'")
    #focal-probiou
    modulating_factor = bc.pow(gamma)
    # modulating_factor = (1 - bc).pow(gamma)
    loss = -modulating_factor * torch.log(bc + eps)
    #kl散度版probiou
    # loss = modulating_factor - torch.log(bc + eps)
    return loss

def laplace_eiou_loss(pred, target, lambda_kl=0.5, eps=1e-6):
    """
    EIoU + Laplace symmetric KL 散度损失（KL作为EIoU的系数形式）。
    pred, target: [N, 4] (x, y, w, h)
    """
    # EIoU 部分（从原 bbox_iou 复制逻辑）
    x1, y1, w1, h1 = pred.unbind(-1)
    x2, y2, w2, h2 = target.unbind(-1)
   
    eiou = bbox_iou(pred, target, xywh=True, EIoU=True)
    eiou_loss = 1.0 - eiou.clamp(min=-1.0, max=1.0)  # 损失形式
   
    # Laplace KL 部分（改写为系数）
    b_x1 = w1 / math.sqrt(24.0) + eps
    b_y1 = h1 / math.sqrt(24.0) + eps
    b_w1 = w1 / math.sqrt(24.0) + eps  # 对 w,h 也用类似尺度
    b_h1 = h1 / math.sqrt(24.0) + eps
   
    b_x2 = w2 / math.sqrt(24.0) + eps
    b_y2 = h2 / math.sqrt(24.0) + eps
    b_w2 = w2 / math.sqrt(24.0) + eps
    b_h2 = h2 / math.sqrt(24.0) + eps
   
    kl_x = symmetric_kl_laplace(x1, x2, b_x1, b_x2, eps)
    kl_y = symmetric_kl_laplace(y1, y2, b_y1, b_y2, eps)
    kl_w = symmetric_kl_laplace(w1, w2, b_w1, b_w2, eps)
    kl_h = symmetric_kl_laplace(h1, h2, b_h1, b_h2, eps)
   
    kl_term = (kl_x + kl_y + kl_w + kl_h) / 4.0  # 平均
   
    loss = eiou_loss * (1.0 + lambda_kl * kl_term)
   
    return loss.clamp(0.0, 100.0)  # 最终保护

class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max=16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.reg_max = reg_max
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
        self.use_wiseiou = True
        self.use_probiou = False
        self.use_laprobiou = False
        if self.use_wiseiou:
            self.wiou_loss = WiseIouLoss(ltype='CIoU', monotonous=True, inner_iou=False, focaler_iou=False)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        if self.use_wiseiou:
            wiou = self.wiou_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask], ret_iou=False, ratio=0.7, d=0.0, u=0.95).unsqueeze(-1)
            # wiou = self.wiou_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask], ret_iou=False, ratio=0.7, d=0.0, u=0.95, **{'scale':0.0}).unsqueeze(-1) # Wise-ShapeIoU,Wise-Inner-ShapeIoU,Wise-Focaler-ShapeIoU
            # wiou = self.wiou_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask], ret_iou=False, ratio=0.7, d=0.0, u=0.95, **{'mpdiou_hw':mpdiou_hw[fg_mask]}).unsqueeze(-1) # Wise-MPDIoU,Wise-Inner-MPDIoU,Wise-Focaler-MPDIoU
            loss_iou = (wiou * weight).sum() / target_scores_sum
        elif self.use_probiou or self.use_laprobiou:
            # ProbIoU 需要 pred 和 target 都是 xywh 格式（大多数 YOLO 实现中 pred_bboxes 是 xyxy）
            # 这里假设需要转换（根据你的 bbox_iou 是否 xywh=False）
            # 如果你的 pred_bboxes / target_bboxes 已经是 xywh，可跳过转换
            pred_xywh = xyxy2xywh(pred_bboxes[fg_mask])   # 需要定义 xyxy2xywh
            targ_xywh = xyxy2xywh(target_bboxes[fg_mask])
            # 调用 probiou_loss（假设它返回 [num_fg] 的损失值）
            if self.use_probiou:
                probiou = probiou_loss(
                    pred=pred_xywh,
                    target=targ_xywh,
                    # mode='l1',
                    gamma=0.5
                )  # shape: [num_fg]
            elif self.use_laprobiou:
                probiou = laplace_eiou_loss(
                    pred=pred_xywh,
                    target=targ_xywh,
                    lambda_kl=0.5
                )
            loss_iou = (probiou.unsqueeze(-1) * weight).sum() / target_scores_sum
        else:
            #原版iou loss
            # iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, EIoU=True) 
            # loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum 
            #仿focal-eiou loss
            iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True,Focal=False,gamma=0.5)
        
            if type(iou) is tuple:
                if len(iou) == 2:
                    loss_iou = ((1.0 - iou[0]) * iou[1].detach() * weight).sum() / target_scores_sum
                else:
                    loss_iou = (iou[0] * iou[1] * weight).sum() / target_scores_sum
            else:
                loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
            
        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class KeypointLoss(nn.Module):
    """Criterion class for computing training losses."""

    def __init__(self, sigmas) -> None:
        """Initialize the KeypointLoss class."""
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class v8DetectionLoss:
    """Criterion class for computing training losses.
    这个是原版损失函数
"""

    def __init__(self, model, tal_topk=10):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none") #用bce
        # self.bce = VarifocalLoss_YOLO(alpha=0.75, gamma=2.0) #用varifocal loss, vfl
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device
        self.use_dfl = m.reg_max > 1
        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

        # # DFL-CIoU gradient probe. It only records statistics and does not change the training loss.
        # self.grad_probe_step = 0
        # self.grad_probe_enabled = os.getenv("YOLO_DFL_CIOU_GRAD_PROBE", "1") == "1"
        # self.grad_probe_interval = max(int(os.getenv("YOLO_DFL_CIOU_GRAD_INTERVAL", "10")), 1)
        # rank = os.getenv("RANK", "0")
        # self.grad_probe_file = os.getenv("YOLO_DFL_CIOU_GRAD_FILE", f"dfl_ciou_grad_stats_rank{rank}.csv")

    # def _probe_dfl_ciou_gradients(self, loss_box, loss_dfl, pred_distri, fg_mask):
    #     """Compute DFL/CIoU gradient norm ratio and cosine similarity on bbox distribution logits."""
    #     if not self.grad_probe_enabled:
    #         return

    #     self.grad_probe_step += 1
    #     if self.grad_probe_step % self.grad_probe_interval != 0:
    #         return

    #     if (not self.use_dfl) or (not pred_distri.requires_grad) or (not fg_mask.any()):
    #         return

    #     try:
    #         g_box = torch.autograd.grad(
    #             loss_box,
    #             pred_distri,
    #             retain_graph=True,
    #             create_graph=False,
    #             allow_unused=True,
    #         )[0]
    #         g_dfl = torch.autograd.grad(
    #             loss_dfl,
    #             pred_distri,
    #             retain_graph=True,
    #             create_graph=False,
    #             allow_unused=True,
    #         )[0]

    #         if g_box is None or g_dfl is None:
    #             return

    #         # Bbox and DFL losses are only computed on positive anchors, so compare positive-anchor gradients only.
    #         g_box_pos = g_box[fg_mask].detach().float().reshape(-1)
    #         g_dfl_pos = g_dfl[fg_mask].detach().float().reshape(-1)

    #         if g_box_pos.numel() == 0 or g_dfl_pos.numel() == 0:
    #             return

    #         box_norm = g_box_pos.norm(p=2)
    #         dfl_norm = g_dfl_pos.norm(p=2)
    #         norm_ratio = dfl_norm / (box_norm + 1e-12)
    #         cosine = F.cosine_similarity(g_box_pos, g_dfl_pos, dim=0, eps=1e-12)

    #         write_header = not os.path.exists(self.grad_probe_file)
    #         with open(self.grad_probe_file, "a", encoding="utf-8") as f:
    #             if write_header:
    #                 f.write(
    #                     "step,box_grad_norm,dfl_grad_norm,dfl_box_norm_ratio,grad_cosine,fg_num,loss_box,loss_dfl\n"
    #                 )
    #             f.write(
    #                 f"{self.grad_probe_step},"
    #                 f"{box_norm.item():.8e},"
    #                 f"{dfl_norm.item():.8e},"
    #                 f"{norm_ratio.item():.8e},"
    #                 f"{cosine.item():.8e},"
    #                 f"{int(fg_mask.sum().item())},"
    #                 f"{float(loss_box.detach().item()):.8e},"
    #                 f"{float(loss_dfl.detach().item()):.8e}\n"
    #             )
    #     except RuntimeError as e:
    #         if os.getenv("YOLO_DFL_CIOU_GRAD_VERBOSE", "0") == "1":
    #             print(f"[DFL-CIoU grad probe skipped] {e}")

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        # dfl_conf = pred_distri.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
        # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2

        # _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
        #     # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
        #     pred_scores.detach().sigmoid(),
        #     (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
        #     anchor_points * stride_tensor,
        #     gt_labels,
        #     gt_bboxes,
        #     mask_gt,
        # )
        target_labels, target_bboxes, target_scores, fg_mask, _ = self.assigner(
                pred_scores.detach().sigmoid(), 
                (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                anchor_points * stride_tensor, 
                gt_labels, 
                gt_bboxes, 
                mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # target_labels = (target_scores > 0).float()
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        # loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE
        if isinstance(self.bce, (nn.BCEWithLogitsLoss, 
                                #  FocalLoss_YOLO
                                 )):
            loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE
        elif isinstance(self.bce, VarifocalLoss_YOLO):
            if fg_mask.sum():
                pos_ious = bbox_iou(pred_bboxes, target_bboxes / stride_tensor, xywh=False).clamp(min=1e-6).detach()
                # 10.0x Faster than torch.one_hot
                cls_iou_targets = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.nc),
                                        dtype=torch.int64,
                                        device=target_labels.device)  # (b, h*w, 80)
                cls_iou_targets.scatter_(2, target_labels.unsqueeze(-1), 1)
                cls_iou_targets = pos_ious * cls_iou_targets
                fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.nc)  # (b, h*w, 80)
                cls_iou_targets = torch.where(fg_scores_mask > 0, cls_iou_targets, 0)
            else:
                cls_iou_targets = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.nc),
                                        dtype=torch.int64,
                                        device=target_labels.device)  # (b, h*w, 80)
            loss[1] = self.bce(pred_scores, cls_iou_targets.to(dtype)).sum() / max(fg_mask.sum(), 1)  # BCE
        elif isinstance(self.bce, 
                        # QualityfocalLoss_YOLO
                        ):
            if fg_mask.sum():
                pos_ious = bbox_iou(pred_bboxes, target_bboxes / stride_tensor, xywh=False).clamp(min=1e-6).detach()
                # 10.0x Faster than torch.one_hot
                targets_onehot = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.nc),
                                        dtype=torch.int64,
                                        device=target_labels.device)  # (b, h*w, 80)
                targets_onehot.scatter_(2, target_labels.unsqueeze(-1), 1)
                cls_iou_targets = pos_ious * targets_onehot
                fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.nc)  # (b, h*w, 80)
                targets_onehot_pos = torch.where(fg_scores_mask > 0, targets_onehot, 0)
                cls_iou_targets = torch.where(fg_scores_mask > 0, cls_iou_targets, 0)
            else:
                cls_iou_targets = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.nc),
                                        dtype=torch.int64,
                                        device=target_labels.device)  # (b, h*w, 80)
                targets_onehot_pos = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.nc),
                                        dtype=torch.int64,
                                        device=target_labels.device)  # (b, h*w, 80)
            loss[1] = self.bce(pred_scores, cls_iou_targets.to(dtype), targets_onehot_pos.to(torch.bool)).sum() / max(fg_mask.sum(), 1)


        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        # self._probe_dfl_ciou_gradients(loss[0], loss[2], pred_distri, fg_mask)

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

# class v8DetectionLoss:
#     """Criterion class for computing training losses.
#     这个是把yolov5的objectness loss加回来的版本，使用bce损失   
#     """

#     def __init__(self, model, tal_topk=10):  # model must be de-paralleled
#         """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
#         device = next(model.parameters()).device  # get model device
#         h = model.args  # hyperparameters

#         m = model.model[-1]  # Detect() module
#         self.bce = nn.BCEWithLogitsLoss(reduction="none")
#         self.hyp = h
#         self.stride = m.stride  # model strides
#         self.nc = m.nc  # number of classes
#         self.no = m.nc + m.reg_max * 4
#         self.reg_max = m.reg_max
#         self.device = device

#         self.use_dfl = m.reg_max > 1

#         # 添加objectness损失相关的初始化
#         self.gr = 0.75  # 用于objectness的iou比率
#         self.sort_obj_iou = False  # 是否对objectness iou排序
#         self.balance = {3: [4.0, 1.0, 0.4]}.get(len(m.stride), [4.0, 1.0, 0.25, 0.06, 0.02])  # 平衡参数
        
#         # Objectness损失函数
#         self.BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h.get('obj_pw', 1.0)], device=device))
        
#         self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
#         self.bbox_loss = BboxLoss(m.reg_max).to(device)
#         self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

#     def preprocess(self, targets, batch_size, scale_tensor):
#         """Preprocesses the target counts and matches with the input batch size to output a tensor."""
#         nl, ne = targets.shape
#         if nl == 0:
#             out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
#         else:
#             i = targets[:, 0]  # image index
#             _, counts = i.unique(return_counts=True)
#             counts = counts.to(dtype=torch.int32)
#             out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
#             for j in range(batch_size):
#                 matches = i == j
#                 if n := matches.sum():
#                     out[j, :n] = targets[matches, 1:]
#             out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
#         return out

#     def bbox_decode(self, anchor_points, pred_dist):
#         """Decode predicted object bounding box coordinates from anchor points and distribution."""
#         if self.use_dfl:
#             b, a, c = pred_dist.shape  # batch, anchors, channels
#             pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
#         return dist2bbox(pred_dist, anchor_points, xywh=False)

#     def __call__(self, preds, batch):
#         """Calculate the sum of the loss for box, cls, dfl and obj multiplied by batch size."""
#         # 增加objectness损失
#         loss = torch.zeros(4, device=self.device)  # box, cls, dfl, obj
#         feats = preds[1] if isinstance(preds, tuple) else preds
        
#         # 恢复原始的预测处理方式
#         pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
#             (self.reg_max * 4, self.nc), 1
#         )

#         pred_scores = pred_scores.permute(0, 2, 1).contiguous()
#         pred_distri = pred_distri.permute(0, 2, 1).contiguous()

#         dtype = pred_scores.dtype
#         batch_size = pred_scores.shape[0]
#         imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
#         anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

#         # Targets
#         targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
#         targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
#         gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
#         mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

#         # Pboxes
#         pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

#         # 使用assigner获取目标信息
#         _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
#             pred_scores.detach().sigmoid(),
#             (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
#             anchor_points * stride_tensor,
#             gt_labels,
#             gt_bboxes,
#             mask_gt,
#         )

#         target_scores_sum = max(target_scores.sum(), 1)

#         # Cls loss
#         loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

#         # Bbox loss
#         if fg_mask.sum():
#             target_bboxes /= stride_tensor
#             loss[0], loss[2] = self.bbox_loss(
#                 pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
#             )

#         # Objectness loss - 修改后的实现
#         # 创建objectness目标
#         tobj = torch.zeros(pred_scores.shape[0], pred_scores.shape[1], 1, 
#                           device=self.device, dtype=pred_scores.dtype)
        
#         if fg_mask.sum():
#             # 计算预测边界框与目标边界框的IoU
#             iou = bbox_iou(pred_bboxes.detach(), target_bboxes, CIoU=True)
            
#             if type(iou) is tuple:
#                 iou = iou[0]
#             iou = iou.squeeze(-1) if iou.dim() > 1 else iou
            
#             # 对正样本设置IoU作为objectness目标
#             iou = iou.detach().clamp(0).type(tobj.dtype)
            
#             # 确保fg_mask和iou的形状匹配
#             if iou.numel() != fg_mask.sum():
#                 # 如果形状不匹配，我们只取正样本的IoU
#                 iou = iou[fg_mask]
            
#             if self.sort_obj_iou:
#                 j = iou.argsort()
#                 iou = iou[j]
#                 # 也需要对fg_mask进行相应的排序
#                 fg_mask_flat = fg_mask.view(-1)
#                 fg_mask_flat = fg_mask_flat.nonzero().squeeze()[j]
#                 fg_mask = torch.zeros_like(fg_mask)
#                 fg_mask.view(-1)[fg_mask_flat] = 1
            
#             if self.gr < 1:
#                 iou = (1.0 - self.gr) + self.gr * iou
            
#             # 将IoU值赋给正样本位置
#             # 使用更简单的方式设置tobj
#             tobj_view = tobj.view(-1)
#             fg_mask_flat = fg_mask.view(-1)
#             tobj_view[fg_mask_flat] = iou
        
#         # 计算objectness损失
#         # 使用分类得分的最大值作为objectness预测的代理
#         pred_obj = pred_scores.sigmoid().max(dim=2, keepdim=True)[0]  # 取每个锚点的最大类别分数
        
#         # 计算objectness损失
#         obj_loss = self.BCEobj(pred_obj.view(-1, 1), tobj.view(-1, 1))
#         loss[3] = obj_loss * self.balance[0]  # 使用第一个平衡权重

#         # 应用超参数权重
#         loss[0] *= self.hyp.box  # box gain
#         loss[1] *= self.hyp.cls  # cls gain
#         loss[2] *= self.hyp.dfl  # dfl gain
#         loss[3] *= self.hyp.get('obj', 0)  # obj gain, 默认为1.0
#         # print('objloss:',loss[3])

#         return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl, obj)

class LogCoshDiceLoss(nn.Module):
    """Log-Cosh Dice损失实现，结合了Dice损失和log-cosh平滑"""
    
    def __init__(self, smooth=1e-3, reduction='mean'):
        """
        初始化Log-Cosh Dice损失
        
        Args:
            smooth: 平滑系数，避免除零错误
            reduction: 损失缩减方式，'mean'或'sum'
        """
        super(LogCoshDiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        计算Log-Cosh Dice损失
        
        Args:
            inputs: 预测值 (logits), 形状为 [N, *]
            targets: 目标值, 形状与inputs相同
            
        Returns:
            log_cosh_dice_loss: Log-Cosh Dice损失值
        """
        # 将logits转换为概率
        inputs = torch.sigmoid(inputs)
        
        # 确保targets与inputs形状一致
        if targets.shape != inputs.shape:
            targets = targets.view_as(inputs)
        
        # 计算交集和并集
        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum()
        
        # 计算Dice系数
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # 标准Dice损失 = 1 - Dice系数
        dice_loss = 1 - dice
        
        # 应用log-cosh变换
        log_cosh_dice_loss = torch.log(torch.cosh(dice_loss))
        
        return log_cosh_dice_loss

# class v8DetectionLoss:
#     """Criterion class for computing training losses.
#     这个是带objectness loss的损失函数，使用log-cosh dice loss
#     """

#     def __init__(self, model, tal_topk=10):  # model must be de-paralleled
#         """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
#         device = next(model.parameters()).device  # get model device
#         h = model.args  # hyperparameters

#         m = model.model[-1]  # Detect() module
#         self.bce = nn.BCEWithLogitsLoss(reduction="none")
#         self.hyp = h
#         self.stride = m.stride  # model strides
#         self.nc = m.nc  # number of classes
#         self.no = m.nc + m.reg_max * 4
#         self.reg_max = m.reg_max
#         self.device = device

#         self.use_dfl = m.reg_max > 1

#         # 添加objectness损失相关的初始化
#         self.gr = 0.75  # 用于objectness的iou比率
#         self.sort_obj_iou = False  # 是否对objectness iou排序
#         self.balance = {3: [4.0, 1.0, 0.4]}.get(len(m.stride), [4.0, 1.0, 0.25, 0.06, 0.02])  # 平衡参数
        
#         # Objectness损失函数 - 使用Log-Cosh Dice损失
#         self.LogCoshDiceObj = LogCoshDiceLoss(smooth=1e-3, reduction='mean')
        
#         self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
#         self.bbox_loss = BboxLoss(m.reg_max).to(device)
#         self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

#     def preprocess(self, targets, batch_size, scale_tensor):
#         """Preprocesses the target counts and matches with the input batch size to output a tensor."""
#         nl, ne = targets.shape
#         if nl == 0:
#             out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
#         else:
#             i = targets[:, 0]  # image index
#             _, counts = i.unique(return_counts=True)
#             counts = counts.to(dtype=torch.int32)
#             out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
#             for j in range(batch_size):
#                 matches = i == j
#                 if n := matches.sum():
#                     out[j, :n] = targets[matches, 1:]
#             out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
#         return out

#     def bbox_decode(self, anchor_points, pred_dist):
#         """Decode predicted object bounding box coordinates from anchor points and distribution."""
#         if self.use_dfl:
#             b, a, c = pred_dist.shape  # batch, anchors, channels
#             pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
#         return dist2bbox(pred_dist, anchor_points, xywh=False)

#     def __call__(self, preds, batch):
#         """Calculate the sum of the loss for box, cls, dfl and obj multiplied by batch size."""
#         # 增加objectness损失
#         loss = torch.zeros(4, device=self.device)  # box, cls, dfl, obj
#         feats = preds[1] if isinstance(preds, tuple) else preds
        
#         # 恢复原始的预测处理方式
#         pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
#             (self.reg_max * 4, self.nc), 1
#         )

#         pred_scores = pred_scores.permute(0, 2, 1).contiguous()
#         pred_distri = pred_distri.permute(0, 2, 1).contiguous()

#         dtype = pred_scores.dtype
#         batch_size = pred_scores.shape[0]
#         imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
#         anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

#         # Targets
#         targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
#         targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
#         gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
#         mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

#         # Pboxes
#         pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

#         # 使用assigner获取目标信息
#         _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
#             pred_scores.detach().sigmoid(),
#             (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
#             anchor_points * stride_tensor,
#             gt_labels,
#             gt_bboxes,
#             mask_gt,
#         )

#         target_scores_sum = max(target_scores.sum(), 1)

#         # Cls loss
#         loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

#         # Bbox loss
#         if fg_mask.sum():
#             target_bboxes /= stride_tensor
#             loss[0], loss[2] = self.bbox_loss(
#                 pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
#             )

#         # Objectness loss - 使用Log-Cosh Dice损失
#         # 创建objectness目标
#         tobj = torch.zeros(pred_scores.shape[0], pred_scores.shape[1], 1, 
#                           device=self.device, dtype=pred_scores.dtype)
        
#         if fg_mask.sum():
#             # 计算预测边界框与目标边界框的IoU
#             iou = bbox_iou(pred_bboxes.detach(), target_bboxes, CIoU=True)
            
#             if type(iou) is tuple:
#                 iou = iou[0]
#             iou = iou.squeeze(-1) if iou.dim() > 1 else iou
            
#             # 对正样本设置IoU作为objectness目标
#             iou = iou.detach().clamp(0).type(tobj.dtype)
            
#             # 确保fg_mask和iou的形状匹配
#             if iou.numel() != fg_mask.sum():
#                 # 如果形状不匹配，我们只取正样本的IoU
#                 iou = iou[fg_mask]
            
#             if self.sort_obj_iou:
#                 j = iou.argsort()
#                 iou = iou[j]
#                 # 也需要对fg_mask进行相应的排序
#                 fg_mask_flat = fg_mask.view(-1)
#                 fg_mask_flat = fg_mask_flat.nonzero().squeeze()[j]
#                 fg_mask = torch.zeros_like(fg_mask)
#                 fg_mask.view(-1)[fg_mask_flat] = 1
            
#             if self.gr < 1:
#                 iou = (1.0 - self.gr) + self.gr * iou
            
#             # 将IoU值赋给正样本位置
#             tobj_view = tobj.view(-1)
#             fg_mask_flat = fg_mask.view(-1)
#             tobj_view[fg_mask_flat] = iou
        
#         # 计算objectness损失
#         # 使用分类得分的最大值作为objectness预测的代理
#         pred_obj = pred_scores.max(dim=2, keepdim=True)[0]  # 取每个锚点的最大类别logits
        
#         # 计算Log-Cosh Dice损失
#         obj_loss = self.LogCoshDiceObj(pred_obj, tobj)
#         loss[3] = obj_loss * self.balance[0]  # 使用第一个平衡权重

#         # 应用超参数权重
#         loss[0] *= self.hyp.box  # box gain 默认7.5
#         loss[1] *= self.hyp.cls  # cls gain 默认0.5
#         loss[2] *= self.hyp.dfl  # dfl gain 默认1.5
#         loss[3] *= self.hyp.get('obj', 0)  # obj gain, 默认为1.0

#         return loss.sum() * batch_size, loss.detach()

class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes the v8SegmentationLoss class, taking a de-paralleled model as argument."""
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n"
                "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolov8n-seg.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'segment' dataset using 'data=coco8-seg.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/segment/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        # loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            # Bbox loss
            loss[0], loss[3] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
            # Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]

            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
            )

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (n, H, W), where n is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (n, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (n, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (n,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
        overlap: bool,
    ) -> torch.Tensor:
        """
        Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8PoseLoss with model, sets keypoint variables and declares a keypoint loss instance."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(
        self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
    ):
        """
        Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            kpts_loss (torch.Tensor): The keypoints loss.
            kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        # Divide coordinates by stride
        selected_keypoints /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses."""

    def __call__(self, preds, batch):
        """Compute the classification loss between predictions and true labels."""
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        loss_items = loss.detach()
        return loss, loss_items


class v8OBBLoss(v8DetectionLoss):
    """Calculates losses for object detection, classification, and box distribution in rotated YOLO models."""

    def __init__(self, model):
        """Initializes v8OBBLoss with model, assigner, and rotated bbox loss; note model must be de-paralleled."""
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    bboxes = targets[matches, 2:]
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
        return out

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
        batch_size = pred_angle.shape[0]  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolov8n-obb.pt data=dota8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xyxy, (b, h*w, 4)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        # Only the first four elements need to be scaled
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    def bbox_decode(self, anchor_points, pred_dist, pred_angle):
        """
        Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).
            pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)


class E2EDetectLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model):
        """Initialize E2EDetectLoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]
