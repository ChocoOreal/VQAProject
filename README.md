This is a Visual question answering system, with SAN[1] as the fusion module, ViT as the image feature extractor, and BERT as the text feature extractor.

To install all dependencies, run
```
pip install -r requirements.txt
```
To run the program, dataset and model's weight need to be downloaded from [here](https://drive.google.com/drive/folders/12CsFC3sE4acnvTziMIC-KGF842isZwPR?usp=sharing). After download completed, move the dataset folder and model.py file to the project root folder.

Run this command to launch the program:

```
streamlit run main.py
```


[1]: Yang, Zichao, et al. "Stacked attention networks for image question answering." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

