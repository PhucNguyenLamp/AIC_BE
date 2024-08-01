import os
import io
import base64
import json

from typing import List, Tuple
import numpy as np

import torch
import open_clip
# from tqdm.notebook import tqdm


from PIL import Image
import faiss
from usearch.index import Index as UsearchIndex
# from matplotlib import pyplot as plt


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


# Factory for creating model and preprocessing
class ModelFactory:
    @staticmethod
    def create_model(model_name: str, device: str):
        try:
            print(f"Attempting to load model on {device}")
            model, _, preprocess = open_clip.create_model_and_transforms(model_name)
            return model.to(device), preprocess
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("GPU out of memory. Falling back to CPU.")
                device = "cpu"
                model, _, preprocess = open_clip.create_model_and_transforms(model_name)
                return model.to(device), preprocess
            else:
                raise e


# Strategy pattern for different indexing methods
class IndexStrategy:
    def build_index(self, embeddings: np.ndarray):
        raise NotImplementedError

    def save_index(self, file_path: str):
        raise NotImplementedError

    def load_index(self, file_path: str):
        raise NotImplementedError

    def search(self, query_embedding: np.ndarray, k: int) -> List[Tuple[int, float]]:
        raise NotImplementedError


class FaissIndexStrategy(IndexStrategy):
    def __init__(self):
        self.faiss_index = None

    def build_index(self, embeddings: np.ndarray):
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(embeddings)

    def save_index(self, file_path: str):
        faiss.write_index(self.faiss_index, file_path)
        print(f"FAISS index saved to {file_path}")

    def load_index(self, file_path: str):
        self.faiss_index = faiss.read_index(file_path)
        print(f"FAISS index loaded from {file_path}")

    def search(self, query_embedding: np.ndarray, k: int) -> List[Tuple[int, float]]:
        faiss.normalize_L2(query_embedding)
        distances, indices = self.faiss_index.search(query_embedding, k)
        return list(zip(indices[0], distances[0]))


class UsearchIndexStrategy(IndexStrategy):
    def __init__(self):
        self.usearch_index = None

    def build_index(self, embeddings: np.ndarray):
        dimension = embeddings.shape[1]
        self.usearch_index = UsearchIndex(ndim=dimension, metric="cosine")
        for i, embedding in enumerate(embeddings):
            self.usearch_index.add(i, embedding)

    def save_index(self, file_path: str):
        self.usearch_index.save(file_path)
        print(f"USearch index saved to {file_path}")

    def load_index(self, file_path: str):
        self.usearch_index = UsearchIndex(ndim=512, metric="cosine")
        self.usearch_index.load(file_path)
        print(f"USearch index loaded from {file_path}")

    def search(self, query_embedding: np.ndarray, k: int) -> List[Tuple[int, float]]:
        matches = self.usearch_index.search(query_embedding, k)
        return [(int(match.key), match.distance) for match in matches]


class CLIPEmbedding:
    def __init__(self, model_name: str, model_nick_name: str, device: str = None):
        self.model_nick_name = model_nick_name
        self.device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model, self.preprocess = ModelFactory.create_model(model_name, self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.faiss_strategy = FaissIndexStrategy()
        self.usearch_strategy = UsearchIndexStrategy()
        self.global_index2image_path = {}

    # def process_image_folder(
    #     self, root_dir: str, output_dir: str, batch_size: int = 32
    # ):
    #     os.makedirs(output_dir, exist_ok=True)

    #     image_paths = self._collect_image_paths(root_dir)
    #     if not image_paths:
    #         print(f"No image found in the given root directory: {root_dir}")
    #         return None

    #     all_embeddings = self._process_images_in_batches(image_paths, batch_size)
    #     if all_embeddings is not None:
    #         self._save_results(all_embeddings, image_paths, output_dir)
    #         return all_embeddings
    #     else:
    #         print("No embeddings were created.")
    #         return None

    # def _collect_image_paths(self, root_dir: str) -> List[str]:
    #     image_paths = []
    #     for root, _, files in os.walk(root_dir):
    #         for file in files:
    #             if file.lower().endswith(
    #                 (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")
    #             ):
    #                 image_paths.append(os.path.join(root, file))
    #     image_paths.sort()
    #     return image_paths

    # def _process_images_in_batches(
    #     self, image_paths: List[str], batch_size: int
    # ) -> np.ndarray:
    #     embeddings = []
    #     for i in tqdm(
    #         range(0, len(image_paths), batch_size),
    #         desc="Processing Batches of images",
    #         unit=f"batch, size = {batch_size}",
    #     ):
    #         batch_paths = image_paths[i : i + batch_size]
    #         batch_embeddings = self._process_batch(batch_paths)
    #         if batch_embeddings is not None:
    #             embeddings.append(batch_embeddings)

    #     if embeddings:
    #         return np.vstack(embeddings)
    #     return None

    # def _process_batch(self, batch_paths: List[str]) -> np.ndarray:
    #     batch_images = []
    #     for img_path in batch_paths:
    #         try:
    #             img = Image.open(img_path).convert("RGB")
    #             img_tensor = self.preprocess(img).unsqueeze(0)
    #             batch_images.append(img_tensor)
    #         except Exception as e:
    #             print(f"Error processing image {img_path}: {str(e)}")
    #             continue

    #     if batch_images:
    #         batch_tensor = torch.cat(batch_images).to(self.device)
    #         with torch.no_grad():
    #             batch_embeddings = (
    #                 self.model.encode_image(batch_tensor)
    #                 .cpu()
    #                 .detach()
    #                 .numpy()
    #                 .astype(np.float32)
    #             )
    #         return batch_embeddings
    #     return None

    # def _save_results(
    #     self, embeddings: np.ndarray, image_paths: List[str], output_dir: str, use_faiss: bool = True
    # ):
    #     clip_file = os.path.join(
    #         output_dir, f"{self.model_nick_name}_clip_embeddings.npy"
    #     )
    #     np.save(clip_file, embeddings)
    #     print(f"CLIP embeddings saved to {clip_file}")

    #     self.global_index2image_path = {i: path for i, path in enumerate(image_paths)}
    #     index_path_file = os.path.join(output_dir, "global2imgpath.json")
    #     with open(index_path_file, "w") as f:
    #         json.dump(self.global_index2image_path, f, indent=4)
    #     print(f"global2imgpath saved to {index_path_file}")

    #     if use_faiss:
    #     # Save FAISS and USearch indexes
    #         self.faiss_strategy.build_index(embeddings)
    #         faiss_file = os.path.join(output_dir, f"{self.model_nick_name}_faiss.bin")
    #         self.faiss_strategy.save_index(faiss_file)
    #     else:
    #         self.usearch_strategy.build_index(embeddings)
    #         usearch_file = os.path.join(output_dir, f"{self.model_nick_name}_usearch.bin")
    #         self.usearch_strategy.save_index(usearch_file)

    async def text_query(
        self, query: str, k: int = 20, use_faiss: bool = True
    ) -> List[Tuple[int, float]]:
        with torch.no_grad():
            text_tokens = self.tokenizer([query]).to(self.device)
            query_embedding = (
                self.model.encode_text(text_tokens)
                .cpu()
                .detach()
                .numpy()
                .astype(np.float32)
            )

        if use_faiss:
            return self.faiss_strategy.search(query_embedding, k)
        else:
            return self.usearch_strategy.search(query_embedding[0], k)

    def image_query(
        self, img_data: str, k: int = 20, use_faiss: bool = True
    ) -> List[Tuple[int, float]]:
        img_bytes = base64.b64decode(img_data)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        img_preprocessed = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            query_embedding = (
                self.model.encode_image(img_preprocessed)
                .cpu()
                .detach()
                .numpy()
                .astype(np.float32)
            )

        if use_faiss:
            return self.faiss_strategy.search(query_embedding, k)
        else:
            return self.usearch_strategy.search(query_embedding[0], k)

    def get_image_paths(self, indices: List[int]) -> List[str]:
        return [self.global_index2image_path.get(i, "Unknown") for i in indices]

    def load_indexes(
        self,
        faiss_path: str = None,
        usearch_path: str = None,
        global2imgpath_path: str = None,
    ):
        if faiss_path:
            self.faiss_strategy.load_index(faiss_path)

        if usearch_path:
            self.usearch_strategy.load_index(usearch_path)

        if global2imgpath_path:
            with open(global2imgpath_path, "r") as f:
                self.global_index2image_path = json.load(f)
        print("Indexes and mappings loaded successfully.")


# def display_result(indices, embedder, k=20):
#     k = min(k, len(indices))

#     n_cols = 3
#     n_rows = (k + n_cols - 1) // n_cols

#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 7 * n_rows))

#     if n_rows == 1:
#         axes = [axes]
#     if n_cols == 1:
#         axes = [[ax] for ax in axes]

#     for i, idx in enumerate(indices[:k]):
#         if i >= k:
#             break

#         row = i // n_cols
#         col = i % n_cols

#         img_path = embedder.get_image_paths([idx])[0]
#         img = Image.open(img_path)
#         axes[row][col].imshow(img)
#         axes[row][col].set_title(f"Rank: {i+1}", fontsize=12)
#         filename = os.path.basename(img_path)
#         axes[row][col].set_xlabel(filename, fontsize=10, wrap=True)

#         axes[row][col].axis("off")

#     for i in range(k, n_rows * n_cols):
#         row = i // n_cols
#         col = i % n_cols
#         axes[row][col].axis("off")

#     plt.tight_layout()
#     plt.subplots_adjust(hspace=0.3, wspace=0.1)
#     plt.show()


# # Initialize the CLIPEmbedding class
# embedder = CLIPEmbedding(
#     model_name="hf-hub:apple/MobileCLIP-B-LT-OpenCLIP",
#     model_nick_name="mobile_clip_B_LT_openCLIP",
# )

# # Load the indexes
# faiss_path = os.path.join(
#     os.path.dirname(__file__),
#     "../../data/embedding/mobile_clip_B_LT_openclip_faiss.bin",
# )
# usearch_path = os.path.join(
#     os.path.dirname(__file__),
#     "../../data/embedding/mobile_clip_B_LT_openclip_usearch.bin",
# )  # Assuming you have a similar file for USearch
# global2imgpath_path = os.path.join(
#     os.path.dirname(__file__), "../../data/embedding/global2imgpath.json"
# )

# # Load the indexes
# embedder.load_indexes(
#     faiss_path=None, usearch_path=usearch_path, global2imgpath_path=global2imgpath_path
# )

# # Perform the search query
# query = "The video shows three Samsung phones at the product launch. Initially, each phone appears one by one and then all three phones appear together."
# print(f"\nOriginal Query: {query}")

# # Measure performance for USearch
# start_time = time.time()
# usearch_results = embedder.text_query(query, k=5, use_faiss=False)
# end_time = time.time()
# print(f"USearch Results: {usearch_results}")
# print(f"USearch Time: {end_time - start_time:.4f} seconds")

# # Optionally, you could test FAISS if you have its index available
# # For demonstration purposes, let's switch the index back to FAISS and re-test
# embedder.load_indexes(
#     faiss_path=faiss_path, usearch_path=None, global2imgpath_path=global2imgpath_path
# )

# # Measure performance for FAISS
# start_time = time.time()
# faiss_results = embedder.text_query(query, k=5, use_faiss=True)
# end_time = time.time()
# print(f"FAISS Results: {faiss_results}")
# print(f"FAISS Time: {end_time - start_time:.4f} seconds")
