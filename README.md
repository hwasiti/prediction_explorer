# Here is what you need to explore your classifier model's prediction

 The function `image_matrix_draw` is going to associate two groups of images to each prediciton of your classifier's validation set predicitons. Basically when you train a NN image classifier, you want to see how it is performing. One way to check its performance, and why it failed on those wrong images, is to checkup the validation images that have been wronlgy classified and to compare it with the images of the wrong class and with the images of the ground truth class. This notebook is doing just that.
 
 Let's take an example:
 
 
 
 
 
 
 
 
This is a function to explore the prediciton's of a classifier. It's purpose is to compare the image that has been classified from the validation set, with the images that has the same predicted class from the train set, and with the images that has the same class from the ground truth label of the validation image in question.
 
 It will draw 7 columns of images. Each column has 4 images, with size (sz):
    The 1st column is composed of 4 replicates of the validation image that we want to explore.
    The 2nd to 4th column composed of images that has the same predicted class from the train set.
    The 5th to 7th column composed of images that has the same class from the ground truth label of the validation image in question.
    
    Parameters:
    val_image_df    :   pandas series that has the following information about the validation image:
                            Image: file name
                            Id   : ground truth label 
                            nbs  : predicted label
                            d    : score
    train_df        :   pandas dataframe of the whole dataset's labels (train + validation) contain the following columns:
                            Image: file name
                            Id   : label
    SZ              :   Size of each image
    MRGN            :   Size of the margin between the major 3 columns (validation column + predicted class images column, ground truth class images column)
    SMALL_MRGN      :   Size of the margin between the each columns of the 7 columns
    BKGR_COLOR      :   Background color 
    HDR             :   Height of the header of the total image
    INPUT_PATH      :   Path of the whole dataset images
    Font            :   Font of the written text 
    FNT_SZ          :   Font size
