# IJDL 2025

This repository is for the paper accepted at the [International Journal on Digital Libraries (IJDL)](https://link.springer.com/journal/799) journal, titled "Searching Agricultural Learning Experiences in the Metaverse via Textual and Visual Queries within the AgriMus project".

In this work, we present a metaverse agricultural museum containing educational video content.

## Dataset

The dataset can be accessed through [Museums file](https://github.com/aliabdari/AgriMus/blob/main/IJDL_2025/final_museums.json).

As mentioned the museums contain videos of agricultural educational information. The videos have been obtained through the [HowTo100M](https://www.di.ens.fr/willow/research/howto100m/). 
The id of the used videos can be found in the prepared dataset. However, the frames of the videos which are used in different procedures of the work can be found the [extracted_frames](https://github.com/aliabdari/AgriMus/tree/main/IJDL_2025/extracted_frames) folder.

## Feature Generation
In this work, in order to represent the museums, we have used different foundational models for the visual and textual representations, like Open CLIP, Mobile CLIP, Blip, Clip4Clip, VideoMAE, and ViViT. The entire implementation for obtaining the embeddings has been provided at the [feature generation](https://github.com/aliabdari/AgriMus/tree/main/IJDL_2025/feature_generation) folder.
 
## Evaluation
In this project, we make use of the  Zero Shot Approach. To reproduce the reported results in the paper, we have provided the implementations in the [evaluation](https://github.com/aliabdari/AgriMus/tree/main/IJDL_2025/evaluation) folder. 

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/aliabdari/AgriMus/blob/main/LICENSE) file for details.



