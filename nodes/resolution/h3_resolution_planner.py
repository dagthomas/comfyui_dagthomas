# H3 Resolution Planner (Crop Only)
#
# Original node and algorithm by **gabbo**.
# Ported into comfyui_dagthomas with the planning logic kept intact; only the
# ComfyUI plumbing (category, tooltips, English messages, plan_info output)
# was adapted to match this pack's conventions.

import math

from ...utils.constants import CUSTOM_CATEGORY


class H3ResolutionPlannerCropOnly:
    """
    H3 Resolution Planner (Crop Only) - by gabbo

    Plans a two-stage generate-then-upscale resolution pair and center-crops the
    input image to the exact aspect ratio of that plan, so no resampling or
    padding is needed anywhere in the chain.

    Stage 1 is the generation size, stage 2 is stage 1 multiplied by the chosen
    upscale factor. Step sizes are picked so that both stages land on clean
    multiples of 32:
      - 2x   -> stage 1 steps of 32, stage 2 steps of 64
      - 1.5x -> stage 1 steps of 64, stage 2 steps of 96
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),

                "resolution_mode": (
                    [
                        "target_megapixels",
                        "max_stage1_from_input",
                        "max_final_from_input",
                    ],
                    {
                        "default": "target_megapixels",
                        "tooltip": (
                            "target_megapixels: hit stage1_megapixels while "
                            "staying close to the input aspect ratio.\n"
                            "max_stage1_from_input: largest stage 1 the input "
                            "can feed within max_crop_percent.\n"
                            "max_final_from_input: largest stage 2 (final) the "
                            "input can feed within max_crop_percent."
                        ),
                    }
                ),

                "stage1_megapixels": ("FLOAT", {
                    "default": 0.40,
                    "min": 0.05,
                    "max": 4.00,
                    "step": 0.01,
                    "tooltip": "Target stage 1 size in megapixels. Only used by target_megapixels mode.",
                }),

                "upscale_mode": (
                    ["2x", "1.5x"],
                    {
                        "default": "2x",
                        "tooltip": "Stage 1 -> stage 2 factor. 1.5x forces stage 1 onto multiples of 64.",
                    }
                ),

                "max_crop_percent": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.0,
                    "max": 25.0,
                    "step": 0.1,
                    "tooltip": (
                        "Maximum share of the input area allowed to be cropped away. "
                        "Only used by the two max_* modes; if nothing fits, the "
                        "least-lossy candidate is used instead."
                    ),
                }),
            }
        }

    RETURN_TYPES = (
        "IMAGE",
        "INT",
        "INT",
        "INT",
        "INT",
        "FLOAT",
        "STRING",
    )

    RETURN_NAMES = (
        "cropped_image",
        "stage1_width",
        "stage1_height",
        "stage2_width",
        "stage2_height",
        "upscale_factor",
        "plan_info",
    )

    FUNCTION = "plan"
    CATEGORY = f"{CUSTOM_CATEGORY}/Resolution"
    DESCRIPTION = (
        "Plans stage 1 / stage 2 resolutions and center-crops the input to that "
        "exact aspect ratio, so the whole chain stays on clean multiples of 32. "
        "Node and algorithm by gabbo."
    )

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------

    def _get_stage1_step_and_scale(self, upscale_mode):
        # For 1.5x, stage1 must be multiple of 64 so stage2 remains
        # a multiple of 32.
        if upscale_mode == "1.5x":
            return 64, 1.5
        return 32, 2.0

    def _get_stage2_step(self, upscale_mode):
        # If stage1 is multiple of 32 and scale is 2x, stage2 is multiple of 64.
        # If stage1 is multiple of 64 and scale is 1.5x, stage2 is multiple of 96.
        if upscale_mode == "1.5x":
            return 96
        return 64

    def _stage2_size(self, w1, h1, upscale_mode):
        if upscale_mode == "1.5x":
            return int(w1 * 3 / 2), int(h1 * 3 / 2)
        return w1 * 2, h1 * 2

    def _stage1_from_stage2(self, w2, h2, upscale_mode):
        if upscale_mode == "1.5x":
            return int(w2 * 2 / 3), int(h2 * 2 / 3)
        return int(w2 / 2), int(h2 / 2)

    def _largest_exact_crop(self, in_w, in_h, target_w, target_h):
        g = math.gcd(int(target_w), int(target_h))
        rw = int(target_w) // g
        rh = int(target_h) // g

        k = min(in_w // rw, in_h // rh)
        if k < 1:
            return None

        crop_w = rw * k
        crop_h = rh * k
        crop_x = max(0, (in_w - crop_w) // 2)
        crop_y = max(0, (in_h - crop_h) // 2)

        return crop_x, crop_y, crop_w, crop_h

    def _crop_loss(self, in_w, in_h, crop_w, crop_h):
        input_area = max(1, in_w * in_h)
        crop_area = crop_w * crop_h
        return 1.0 - (crop_area / input_area)

    # ------------------------------------------------------------
    # Mode 1: target MP
    # ------------------------------------------------------------

    def _find_target_mp_resolution(
        self,
        in_w,
        in_h,
        stage1_megapixels,
        upscale_mode,
        search_radius_steps=12
    ):
        target_ratio = in_w / in_h
        target_pixels = stage1_megapixels * 1024 * 1024

        step1, _ = self._get_stage1_step_and_scale(upscale_mode)

        ideal_w = math.sqrt(target_pixels * target_ratio)
        ideal_h = math.sqrt(target_pixels / target_ratio)

        center_w = max(step1, round(ideal_w / step1) * step1)
        center_h = max(step1, round(ideal_h / step1) * step1)

        best = None
        best_score = float("inf")

        for wi in range(-search_radius_steps, search_radius_steps + 1):
            for hi in range(-search_radius_steps, search_radius_steps + 1):
                w1 = center_w + wi * step1
                h1 = center_h + hi * step1

                if w1 < step1 or h1 < step1:
                    continue

                w2, h2 = self._stage2_size(w1, h1, upscale_mode)

                if (w2 % 32) != 0 or (h2 % 32) != 0:
                    continue

                crop = self._largest_exact_crop(in_w, in_h, w1, h1)
                if crop is None:
                    continue

                ratio = w1 / h1
                aspect_err = abs(math.log(ratio / target_ratio))
                pixel_err = abs((w1 * h1) - target_pixels) / target_pixels
                crop_loss = self._crop_loss(in_w, in_h, crop[2], crop[3])

                score = aspect_err * 5.0 + pixel_err + crop_loss * 0.25

                if score < best_score:
                    best_score = score
                    best = (w1, h1, w2, h2, crop)

        if best is None:
            raise ValueError(
                f"No valid resolution found for a {in_w}x{in_h} input at "
                f"{stage1_megapixels:.2f} MP ({upscale_mode}). "
                "Try a different target or upscale mode."
            )

        return best

    # ------------------------------------------------------------
    # Mode 2: max Stage 1 from input
    # ------------------------------------------------------------

    def _find_max_stage1_from_input(
        self,
        in_w,
        in_h,
        upscale_mode,
        max_crop_percent
    ):
        step1, _ = self._get_stage1_step_and_scale(upscale_mode)
        max_crop_loss = max_crop_percent / 100.0

        valid = []
        fallback = []

        for w1 in range(step1, in_w + 1, step1):
            for h1 in range(step1, in_h + 1, step1):
                w2, h2 = self._stage2_size(w1, h1, upscale_mode)

                if (w2 % 32) != 0 or (h2 % 32) != 0:
                    continue

                crop = self._largest_exact_crop(in_w, in_h, w1, h1)
                if crop is None:
                    continue

                _, _, crop_w, crop_h = crop

                # Stage 1 canvas must fit within the cropped source.
                if w1 > crop_w or h1 > crop_h:
                    continue

                loss = self._crop_loss(in_w, in_h, crop_w, crop_h)
                area = w1 * h1

                item = (area, -loss, w1, h1, w2, h2, crop)
                fallback.append(item)

                if loss <= max_crop_loss + 1e-12:
                    valid.append(item)

        pool = valid if valid else fallback
        if not pool:
            raise ValueError(
                f"No valid resolution found for a {in_w}x{in_h} input "
                f"({upscale_mode}). The image is likely smaller than one "
                f"{step1}px step."
            )

        if valid:
            best = max(pool, key=lambda x: (x[0], x[1]))
        else:
            best = max(pool, key=lambda x: (x[1], x[0]))

        _, _, w1, h1, w2, h2, crop = best
        return w1, h1, w2, h2, crop, not valid

    # ------------------------------------------------------------
    # Mode 3: max FINAL from input
    # ------------------------------------------------------------

    def _find_max_final_from_input(
        self,
        in_w,
        in_h,
        upscale_mode,
        max_crop_percent
    ):
        step2 = self._get_stage2_step(upscale_mode)
        step1, _ = self._get_stage1_step_and_scale(upscale_mode)
        max_crop_loss = max_crop_percent / 100.0

        valid = []
        fallback = []

        for w2 in range(step2, in_w + 1, step2):
            for h2 in range(step2, in_h + 1, step2):
                w1, h1 = self._stage1_from_stage2(w2, h2, upscale_mode)

                if w1 < step1 or h1 < step1:
                    continue
                if (w1 % step1) != 0 or (h1 % step1) != 0:
                    continue

                crop = self._largest_exact_crop(in_w, in_h, w2, h2)
                if crop is None:
                    continue

                _, _, crop_w, crop_h = crop

                # Final canvas must fit within the cropped source.
                if w2 > crop_w or h2 > crop_h:
                    continue

                loss = self._crop_loss(in_w, in_h, crop_w, crop_h)
                area = w2 * h2

                item = (area, -loss, w1, h1, w2, h2, crop)
                fallback.append(item)

                if loss <= max_crop_loss + 1e-12:
                    valid.append(item)

        pool = valid if valid else fallback
        if not pool:
            raise ValueError(
                f"No valid resolution found for a {in_w}x{in_h} input "
                f"({upscale_mode}). The image is likely smaller than one "
                f"{step2}px final step."
            )

        if valid:
            best = max(pool, key=lambda x: (x[0], x[1]))
        else:
            best = max(pool, key=lambda x: (x[1], x[0]))

        _, _, w1, h1, w2, h2, crop = best
        return w1, h1, w2, h2, crop, not valid

    # ------------------------------------------------------------
    # Main
    # ------------------------------------------------------------

    def plan(
        self,
        image,
        resolution_mode,
        stage1_megapixels,
        upscale_mode,
        max_crop_percent
    ):
        if len(image.shape) != 4:
            raise ValueError(f"Unexpected image format: {tuple(image.shape)}")

        _, in_h, in_w, _ = image.shape
        in_w = int(in_w)
        in_h = int(in_h)

        # True when a max_* mode could not satisfy max_crop_percent and had to
        # settle for the least-lossy candidate instead.
        over_budget = False

        if resolution_mode == "max_stage1_from_input":
            stage1_w, stage1_h, stage2_w, stage2_h, crop, over_budget = (
                self._find_max_stage1_from_input(
                    in_w=in_w,
                    in_h=in_h,
                    upscale_mode=upscale_mode,
                    max_crop_percent=max_crop_percent
                )
            )
        elif resolution_mode == "max_final_from_input":
            stage1_w, stage1_h, stage2_w, stage2_h, crop, over_budget = (
                self._find_max_final_from_input(
                    in_w=in_w,
                    in_h=in_h,
                    upscale_mode=upscale_mode,
                    max_crop_percent=max_crop_percent
                )
            )
        else:
            stage1_w, stage1_h, stage2_w, stage2_h, crop = (
                self._find_target_mp_resolution(
                    in_w=in_w,
                    in_h=in_h,
                    stage1_megapixels=stage1_megapixels,
                    upscale_mode=upscale_mode,
                    search_radius_steps=12
                )
            )

        crop_x, crop_y, crop_w, crop_h = crop

        cropped = image[
            :,
            crop_y:crop_y + crop_h,
            crop_x:crop_x + crop_w,
            :
        ]

        _, scale = self._get_stage1_step_and_scale(upscale_mode)

        info = self._format_plan_info(
            in_w=in_w,
            in_h=in_h,
            crop=crop,
            stage1_w=stage1_w,
            stage1_h=stage1_h,
            stage2_w=stage2_w,
            stage2_h=stage2_h,
            resolution_mode=resolution_mode,
            upscale_mode=upscale_mode,
            max_crop_percent=max_crop_percent,
            over_budget=over_budget,
        )

        return (
            cropped,
            int(stage1_w),
            int(stage1_h),
            int(stage2_w),
            int(stage2_h),
            float(scale),
            info,
        )

    def _format_plan_info(
        self,
        in_w,
        in_h,
        crop,
        stage1_w,
        stage1_h,
        stage2_w,
        stage2_h,
        resolution_mode,
        upscale_mode,
        max_crop_percent,
        over_budget,
    ):
        crop_x, crop_y, crop_w, crop_h = crop
        loss_pct = self._crop_loss(in_w, in_h, crop_w, crop_h) * 100.0
        g = math.gcd(int(stage1_w), int(stage1_h))

        lines = [
            f"mode: {resolution_mode} @ {upscale_mode}",
            f"input: {in_w}x{in_h}",
            f"crop: {crop_w}x{crop_h} at ({crop_x},{crop_y}) - {loss_pct:.2f}% of area removed",
            f"aspect: {int(stage1_w) // g}:{int(stage1_h) // g}",
            f"stage 1: {stage1_w}x{stage1_h} ({(stage1_w * stage1_h) / 1048576.0:.2f} MP)",
            f"stage 2: {stage2_w}x{stage2_h} ({(stage2_w * stage2_h) / 1048576.0:.2f} MP)",
        ]

        # The max_* modes silently fall back to the least-lossy candidate when
        # nothing fits the budget, so say when that happened.
        if over_budget:
            lines.append(
                f"WARNING: no plan fit within max_crop_percent ({max_crop_percent:.1f}%); "
                f"used the least-lossy option at {loss_pct:.2f}%."
            )

        return "\n".join(lines)


NODE_CLASS_MAPPINGS = {
    "H3ResolutionPlannerCropOnly": H3ResolutionPlannerCropOnly,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ResolutionPlannerCropOnly": "APNext H3 Resolution Planner (Crop Only) - by gabbo",
}
