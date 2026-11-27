"""Each validation image's metrics, which simple_trainer only averages."""

import csv

METRICS = ("psnr", "ssim", "lpips")


def record_per_image_metrics(runner_class):
    """Make Runner.eval also write stats/<stage>_step<step>_per_image.csv.

    eval scores every image with self.psnr, self.ssim, and self.lpips, then
    the color-corrected render with them again when use_color_correction_metric
    is set, so each call's value is recorded on the way through.
    """
    original_eval = runner_class.eval

    def evaluate(self, step, stage="val"):
        metrics = {name: getattr(self, name) for name in METRICS}
        values = {name: [] for name in METRICS}
        for name, metric in metrics.items():
            def record(*args, metric=metric, recorded=values[name]):
                value = metric(*args)
                recorded.append(float(value))
                return value
            setattr(self, name, record)
        try:
            original_eval(self, step, stage=stage)
        finally:
            for name, metric in metrics.items():
                setattr(self, name, metric)
        if self.world_rank != 0:
            return
        color_corrected = self.cfg.use_color_correction_metric
        columns = {name: values[name][::2 if color_corrected else 1] for name in METRICS}
        if color_corrected:
            columns.update({f"cc_{name}": values[name][1::2] for name in METRICS})
        names = [self.parser.image_names[index] for index in self.valset.indices]
        with open(f"{self.stats_dir}/{stage}_step{step:04d}_per_image.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["image", *columns])
            for row in zip(names, *columns.values(), strict=True):
                writer.writerow([row[0], *(f"{value:.6g}" for value in row[1:])])

    runner_class.eval = evaluate
