An attempt to learn various image processing as well as modern well established models for detecting violation of not wearing a helmet and to localise and extract the number plate from that.

Logs : 
Starting up with the License plate localisation part along with segmenting the license plate into individual characters and detecting the characters individually to output the license number as text.
* Build up an OCR model with a dataset containing 36000 images belonging to classes A-Z and 0-9.
* Starting up with the localisation part.
    * Image processing approach (still not started)
    * Use models to give bounding boxes -- Mask RCNN
        * Explored Labelme and CVAT for labelling and annotating.
        * Time consuming to manually annotate the entire dataset.
        * Downloading and using pretrained models like Detectron2 by Meta AI to automate the annotation process.Finally manual scrutiny and correction of bounding boxes using CVAT before training.
             * pip install 'git+https://github.com/facebookresearch/detectron2.git'
             * Uses a ResNet-50 backbone with a Feature Pyramid Network (FPN).
             * Is pre-trained on COCO dataset for instance segmentation (detecting objects + drawing masks).
                
        * Training a Mask RCNN to detect license plates for new images.
