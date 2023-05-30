import io
import base64
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict


class Processor:
    def __init__(self, path_to_gt):
        self.gt_df = pd.read_parquet(path_to_gt)

    def __call__(self, submission_df):
        submission_df = Processor.to_nice_formatting(submission_df)
        submission_df = submission_df.merge(right=self.gt_df, on="id")
        similarities = submission_df.apply(Processor.compute_cosine_similarity, axis=1)
        similarities = pd.Series(np.array([el[1] for el in similarities]))

        metrics = similarities.describe().to_dict()
        sims_hist_img = Processor.plot_sims_hist(similarities, metrics)
        intersection_percent = round(100 * len(submission_df) / len(self.gt_df), 3)

        return {
            "row_count": int(metrics["count"]),
            "mean_similarity": round(metrics["mean"], 3),
            "std_similarity": round(metrics["std"], 3),
            "median_similarity": round(metrics["50%"], 3),
            "intersection_percent": intersection_percent,
            "sims_histplot": sims_hist_img,
        }

    @staticmethod
    def plot_sims_hist(similarities: pd.Series, metrics: Dict[str, float]) -> str:
        plt.hist(similarities, bins=100)
        plt.title("Гистограмма Cosine Similarity")
        plt.axvline(x=metrics["mean"], color="red", label="mean")
        plt.yticks([])
        plt.legend()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)

        image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return image_base64

    @staticmethod
    def cosine_similarity(vec1, vec2):
        vec1, vec2 = np.array(vec1), np.array(vec2)
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)

    @staticmethod
    def compute_cosine_similarity(row):
        return row["id"], Processor.cosine_similarity(row["emb_x"], row["emb_y"])

    @staticmethod
    def to_nice_formatting(input_df) -> pd.DataFrame:
        result = []
        curr_res = []
        curr_id = None
        for img_id, val in tqdm(input_df.values):
            real_id, *_ = img_id.split("_")
            if curr_id is None:
                curr_id = real_id
            if real_id == curr_id:
                curr_res.append(val)
            else:
                assert len(curr_res) == 384
                result.append((curr_id, curr_res))
                curr_res = [val]
                curr_id = real_id
        return pd.DataFrame(result, columns=["id", "emb"])
