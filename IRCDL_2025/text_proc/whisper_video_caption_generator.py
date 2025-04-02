import os
from tqdm import tqdm
import whisper
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
import pickle


def extract_transcribe_v2(vid, model_):
    return model_.transcribe(vid)


def extract_transcribe(vid):
    model = whisper.load_model("base")  # Options: tiny, base, small, medium, large
    return model.transcribe(vid)


def process_batch_videos(videos_batch, n_jobs=10):
    # Parallelize over each point cloud in the batch
    with tqdm_joblib(tqdm(desc="Processing Videos", total=len(videos_batch))) as progress_bar:
        videos_batch = Parallel(n_jobs=n_jobs)(
            delayed(extract_transcribe)(vid) for vid in videos_batch
        )
    return videos_batch


if __name__ == '__main__':
    videos_path = 'dataset/videos'
    videos = [videos_path + os.sep + vid for vid in os.listdir(videos_path)]
    # results = process_batch_videos(videos, n_jobs=-1)
    model = whisper.load_model("base", device="cuda")
    results = [extract_transcribe_v2(vid, model) for vid in tqdm(videos)]
    with open('transcriptions.pkl', 'wb') as file:
        pickle.dump(results, file)
    print(len(results))



