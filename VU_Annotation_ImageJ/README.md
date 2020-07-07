# Human Annotation Analysis using ImageJ

## Datasets
- VAN0003-LK-32-21-PAS_registered.ome.tif
- VAN0006-LK-2-85-PAS_registered.ome.tif
- VAN0008-RK-403-100-PAS_registered.ome.tif
- VAN0011-RK-3-10-PAS_registered.ome.tif
- VAN0012-RK-103-75-PAS_registered.ome.tif
- VAN0013-LK-202-96-PAS_registered.ome

## Workflow for Annotation using ImageJ
- Manually annotate slides using ImageJ
- Create a label for each annotated glomeruli as ‘Glomeruli’
- For each annotation, .roi object will be generated
- After saving the annotation of corresponding slides, a zip file will be created for each slide having respective .roi objects for each annotation

## Reading .ROIs object in Python
Since the file size of each .tif images are very large, image compression is done by 50%(without losing the resolution) and then converted into jpg format in order to read the file in Python. The annotation of Glomeruli was done using the ‘oval’ tool. After reading the images, the corresponding .roi zip file is also read using read_roi library in Python. The variable contains each annotation of Glomeruli as a key value pair where key is the roi name and value contains :
Type : {oval,polygon, circle, box, etc.)
Left : Position of leftmost region
Top : Position of topmost region
Width : Horizontal spread
Height : Vertical spread
Name : Label name

In order to draw ellipses over the entire images, Pillow library was used. The Pillow library takes the following argument for position: leftmost, rightmost, topmost and bottommost point. The center points are calculated using the below formula:
Centre(x) = left + width/2
Centre(y) = top + height/2


Then positions were calculated using the below formula:
Leftmost point = Centre(x) - width/2
Rightmost point = Centre(x) - width/2
Topmost point = Centre(y) - height/2
Bottommost point = Centre(y) + height/2

All the glomeruli ovals were drawn by iterating through all the keys present in a single .roi zip file.
 
