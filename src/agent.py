# Import the MIScnn module
import math
import os
import random
from datetime import datetime
import miscnn
import numpy as np
from PIL import Image
from tensorboard import program
import custom_visualizer
from custom_io import NuCLS_IO
from miscnn.processing.subfunctions import Normalization, Clipping, Resampling, Padding
from miscnn.neural_network.architecture.unet.standard import Architecture
from miscnn.neural_network.metrics import dice_crossentropy, tversky_loss, dice_soft, tversky_crossentropy, \
    dice_soft_loss, sym_unified_focal_loss, asym_unified_focal_loss, symmetric_focal_tversky_loss, \
    categorical_focal_loss, combo_loss, focal_tversky_loss, symmetric_focal_loss, asymmetric_focal_loss, \
    asymmetric_focal_tversky_loss
import pandas as pd
from tqdm import tqdm
import tensorflow.keras.callbacks as cb
from net_config import net_config
import seaborn as sns
import miseval


# import Medical Data
def create_configurations():
    configurations = []
    raw_classes = [(253, "fov"), (1, "tumor"), (2, "fibroblast"), (3, "lymphocyte"), (4, "plasma_cell"), (5, "macrophage"), (6, "mitotic_figure"), (7, "vascular_endotheliu"), (8, "myoepithelium"), (9, "apoptotic_body"), (10, "neutrophil"), (11, "ductal_epithelium"), (12, "eosinophil"), (99, "unlabeled")]

    # Network Parameter
    analysis = "patchwise-grid"
    batch_size = 20
    epochs = 10000

    # Medical Data
    paths = []
    # https://sites.google.com/view/nucls/single-rater (corrected) single-rater only
    data_path = os.path.join(os.path.join(os.path.join("..", ".."), "QC-20211203T130409Z-001"), "QC")
    data_path_img = os.path.join(data_path, "rgb")
    data_path_mask = os.path.join(data_path, "mask")
    #paths.append((data_path_img, data_path_mask, "single_rater_only", 0.6, 0.2, 0.2))

    # https://sites.google.com/view/nucls/single-rater (corrected) & https://sites.google.com/view/nucls/multi-rater (Evaluation)
    data_path = os.path.join(os.path.join("..", ".."), "QC_PsAreTruth_merge")
    data_path_img = os.path.join(data_path, "rgb")
    data_path_mask = os.path.join(data_path, "mask")
    paths.append((data_path_img, data_path_mask, "single_rater_train+val--multi_rater_pred", 0.7, 0.27, 0.0305))

    # https://sites.google.com/view/nucls/single-rater (corrected) & https://sites.google.com/view/nucls/multi-rater (Evaluation)
    data_path = os.path.join(os.path.join("..", ".."), "QC_PsAreTruth_merge")
    data_path_img = os.path.join(data_path, "rgb")
    data_path_mask = os.path.join(data_path, "mask")
    #paths.append((data_path_img, data_path_mask, "single_train--multi_val+pred", 0.97, 0.015, 0.015))

    # https://sites.google.com/view/nucls/single-rater (corrected) & https://sites.google.com/view/nucls/multi-rater (Evaluation) & BCCS merge
    data_path = os.path.join(os.path.join("..", ".."), "QC_BCCS_merge")
    data_path_img = os.path.join(data_path, "rgb")
    data_path_mask = os.path.join(data_path, "mask")
    #paths.append((data_path_img, data_path_mask, "single_BCSS_train--multi_pred", 0.7, 0.272, 0.028))

    # https://github.com/PathologyDataScience/BCSS - not the challenge but same data - 21 classes
    data_path = os.path.join(os.path.join("..", ".."), "0_Public-data-Amgad2019_0.25MPP")
    data_path_img = os.path.join(data_path, "rgb")
    data_path_mask = os.path.join(data_path, "mask")
    #paths.append((data_path_img, data_path_mask, "BCSS", 0.7, 0.15, 0.15))

    # https://sites.google.com/view/nucls/single-rater (corrected) & https://github.com/PathologyDataScience/BCSS (cutted in tiles)
    data_path = os.path.join(os.path.join(os.path.join("..", ".."), "BCCS-tiles_QC_merge"))
    data_path_img = os.path.join(data_path, "rgb")
    data_path_mask = os.path.join(data_path, "mask")
    #paths.append((data_path_img, data_path_mask, "single_rater_only", 0.6, 0.2, 0.2))

    #alpha = [0.000348792, 0.0075, 0.022, 0.0128, 1] -- 100% correct, without boundingboxes
    #alpha = [0.015884227, 0.34111394657, 1, 0.5842755, 1] without boundingboxes
    #alpha = [0.0716436639, 0.334672115446, 0.919219909, 0.0535, 1] #clear data
    #alpha = [0.2, 0.2956, 0.2951, 0.3506, 8.5073]
    alpha = [0.25, 0.37, 0.3663, 0.4343] # without other

    alpha2 = [0.36986, 1]

    losses = [categorical_focal_loss(alpha, 0.25)]
    #losses = [dice_crossentropy, categorical_focal_loss(alpha, 0.25), asymmetric_focal_tversky_loss]
    #labels = ["categorical_focal_loss (0.015884227, 0.34111394657, 1, 0.5842755, 1)", "categorical_focal_loss (0.36986, 1)"]
    #labels = ["dice_crossentropy","categorical_focal_loss", "asymmetric_focal_tversky_loss"]
    labels = ["categorical_focal_loss"]

    for i, loss in enumerate(losses):
        for path in paths:
            config = net_config(path[0], path[1], raw_classes, raw_classes, batch_size, epochs, analysis, loss, path[2]+"--all_classes--"+labels[i], path[3], path[4], path[5])
            # configurations.append(config)

            # super classes
            classes = [(253, "fov"), (1, "tumor"), (2, "Stromal"), (3, "sTILs")]
            config = net_config(path[0], path[1], classes, raw_classes, batch_size, epochs, analysis, loss, path[2]+"--super_classes--"+labels[i], path[3], path[4], path[5])
            configurations.append(config)

            # tumor + fov
            """
            if i == 1:
                # tumor + fov 
                classes = [(253, "fov"), (1, "all other classes")]
                config = net_config(path[0], path[1], classes, raw_classes, batch_size, epochs, analysis, loss, path[2]+"--fov_vs_other_classes--"+labels[i], path[3], path[4], path[5])
                configurations.append(config)
            """
    return configurations

def agent(config, overview_file, date, results_folder_path):
    results_path = os.path.join(results_folder_path, config.info)
    checkpoints_path = os.path.join(results_path, "checkpoints")
    model_path = os.path.join(results_path, "model")
    csv_path = os.path.join(results_path, "csv")
    tensorboard_path = os.path.join(results_path, "tensorboard")
    ground_truth_path = os.path.join(results_path, "ground_truth_check")
    boxplot_path = os.path.join(results_path, "boxplots")

    # Create Log Infrastucture
    mode = 0o666
    print("\n\n---> Create Log Infrastructure\n")
    os.mkdir(results_path, mode)
    os.mkdir(checkpoints_path)
    os.mkdir(model_path)
    os.mkdir(csv_path)
    os.mkdir(tensorboard_path)
    os.mkdir(ground_truth_path)
    os.mkdir(boxplot_path)
    log_file = open(os.path.join(results_path, "log_process.txt"), "w")

    #possible_patch_sizes = [128, 256, 512, 1024, 2048]
    possible_patch_sizes = [256, 512, 1024, 2048]

    # Calculate number of classes
    num_of_classes = len(config.classes)

    # Create the Data I/O object for images
    print("\n---> Create I/O Objects", file=log_file)

    # NuCLS
    interface = NuCLS_IO(config.data_path_img, config.data_path_mask, "RGB", "png", 3, num_of_classes, config.exclude_classes, False)

    # BCCS
    # interface = BCCS_IO(data_path_img, data_path_mask, "RGB", "png", 3, num_of_classes, False)

    data_io = miscnn.Data_IO(interface, config.data_path_img)

    print("\n---> Start Data Exploration\n", file=log_file)
    # data exploration
    sample_list = data_io.get_indiceslist()
    sample_list.sort()
    ground_truth_check = 0
    # load each sample and obtain collect diverse information from them
    sample_data = {}
    for index in tqdm(sample_list):
        # Sample loading
        sample = data_io.sample_loader(index, load_seg=True)
        if ground_truth_check < 10:
            custom_visualizer.create_superclass_ground_truth(sample.img_data, sample.seg_data, ground_truth_path, str(sample.index))
            ground_truth_check += 1
        # Create an empty list for the current sample in our data dictionary
        sample_data[index] = []
        # Store the volume shape
        sample_data[index].append(sample.img_data.shape)
        # Identify minimum and maximum volume intensity
        sample_data[index].append(sample.img_data.min())
        sample_data[index].append(sample.img_data.max())
        # Identify and store class distribution
        unique_data, unique_counts = np.unique(np.squeeze(sample.seg_data, axis=-1), return_counts=True)
        class_freq = unique_counts / np.sum(unique_counts)
        class_freq_onehot = []

        for i, x in enumerate(config.classes):
            _tmp = 0
            if i in unique_data:
                idx = np.where(unique_data == i)[0][0]
                #_tmp = class_freq[idx]
                _tmp = 1
            class_freq_onehot.append(_tmp)
        class_freq = np.around(class_freq_onehot, decimals=6)
        sample_data[index].append(tuple(class_freq))

    # Transform collected data into a pandas dataframe
    df = pd.DataFrame.from_dict(sample_data, orient="index",
                                columns=["vol_shape", "vol_minimum",
                                         "vol_maximum", "class_frequency"])

    print("\n---> Print Image Details\n", file=log_file)
    # Calculate mean and median shape sizes
    shape_list = np.array(df["vol_shape"].tolist())
    patch_shape = []
    for i, a in enumerate(["X", "Y"]):
        print(a + "-Axes Min:", np.min(shape_list[:,i]), file=log_file)
        print(a + "-Axes Max:", np.max(shape_list[:,i]), file=log_file)
        print(a + "-Axes Mean:", np.mean(shape_list[:,i]), file=log_file)
        median = math.floor(int(np.median(shape_list[:, i])))
        patch_size = min(possible_patch_sizes, key=lambda x: abs(x - median))
        patch_shape.append(patch_size)
        print(a + "-Axes Median:", np.median(shape_list[:,i]), file=log_file)
    # Calculate average class frequency
    df_classes = pd.DataFrame(df["class_frequency"].tolist(),
                              columns=config.classes)
    #print(df_classes)
    print("\n---> Print Class Details\n", file=log_file)
    print(df_classes.sum(axis=0), file=log_file)


    # Set Padding-shape
    patch_shape = tuple(patch_shape)
    padding_shape = patch_shape
    #patch_shape = (128, 128)
    #padding_shape = (128, 128)

    # Sample splitting
    num_of_samples = len(sample_list)
    train_start_sample = 0
    train_end_sample = math.floor(num_of_samples * config.train_data_split)
    save_freq_checkpoint = math.floor((train_end_sample - train_start_sample) / config.batch_size)
    validation_start_sample = train_end_sample + 1
    validation_end_sample = validation_start_sample + math.floor(num_of_samples * config.val_data_split)
    predict_start_sample = validation_end_sample + 1
    predict_end_sample = predict_start_sample + math.floor(num_of_samples * config.pred_data_split) - 1

    """
    # Only for Mixed samplesets zb QC + truth
    train_start_sample = 0
    train_end_sample = num_of_samples-53-1
    save_freq_checkpoint = math.floor((train_end_sample - train_start_sample) / config.batch_size)
    validation_start_sample = train_end_sample + 1
    validation_end_sample = validation_start_sample + 30
    predict_start_sample = validation_end_sample + 1
    predict_end_sample = num_of_samples
    """

    print("\n\n---> Create Configuration", file=log_file)
    print("\n--------> Result Paths", file=log_file)
    print("--------> results        : "+results_path, file=log_file)
    print("--------> checkpoints    : "+checkpoints_path, file=log_file)
    print("--------> final model    : "+model_path, file=log_file)
    print("--------> csv files      : "+csv_path, file=log_file)
    print("--------> tensorboard    : "+tensorboard_path, file=log_file)
    print("\n--------> Medical Source", file=log_file)
    print("--------> clean images   : "+config.data_path_img, file=log_file)
    print("--------> seg images     : "+config.data_path_img, file=log_file)
    print("--------> num of classes : "+str(num_of_classes), file=log_file)
    print("--------> num of samples : "+str(num_of_samples), file=log_file)
    print("--------> train spectrum : "+str(train_start_sample)+" - "+str(train_end_sample), file=log_file)
    print("--------> val. spectrum  : "+str(validation_start_sample)+" - "+str(validation_end_sample), file=log_file)
    print("--------> pred. spectrum : "+str(predict_start_sample)+" - "+str(predict_end_sample), file=log_file)
    print("\n--------> Network Config", file=log_file)
    print("--------> analyse type   : "+config.analysis, file=log_file)
    print("--------> patch shape    : "+str(patch_shape), file=log_file)
    print("--------> batch size     : "+str(config.batch_size), file=log_file)
    print("--------> epochs         : "+str(config.epochs), file=log_file)
    print("--------> checkpoint freq: "+str(save_freq_checkpoint), file=log_file)

    print("\n\n---> Initilize Preprocessor\n", file=log_file)
    # Create and configure the Data Augmentation class
    data_aug = miscnn.Data_Augmentation(cycles=1, scaling=True, rotations=True,
                                        elastic_deform=True, mirror=True,
                                        brightness=True, contrast=True,
                                        gamma=True, gaussian_noise=True)

    print("\n\n---> Create Subfunctions\n", file=log_file)
    # Create subfunctions
    normalize_sf = Normalization(mode="z-score")
    padding_sf = Padding(min_size=padding_shape, shape_must_be_divisible_by=2)
    subfunctions = [padding_sf, normalize_sf]

    print("\n\n---> Create Network\n", file=log_file)
    # Create and configure the Preprocessor class
    pp = miscnn.Preprocessor(data_io, batch_size=config.batch_size, data_aug=data_aug, analysis=config.analysis, patch_shape=patch_shape, subfunctions=subfunctions)
    unet_standard = Architecture()
    #unet_standard = custom_architecture()
    # Create a deep learning neural network model with a standard U-Net architecture
    model = miscnn.Neural_Network(preprocessor=pp, architecture=unet_standard,
                                  loss=config.loss, metrics=["accuracy", dice_soft],
                                  learning_rate=0.001)

    # here we could add a model throw the console or with path
    model.load("./results/bounding/single_rater_only--super_classes--categorical_focal_loss/model/model.hdf5")
    """
    print("\n\nYou could import a (already trained) Network.")
    print("Specify the path to the model file (.hdf5) or press only enter for a new model:\n")
    # input
    while 1:
        print("Path or Enter: ")
        string = str(input())
        if os.path.exists(string):
            model.load(string)
            break
        if string == "":
            break
        print("Model not found! Try again.")
    """

    print("\n\n---> Start Training\n", file=log_file)
    # Create a callback that saves the model's weights every epoch
    cb_cp = cb.ModelCheckpoint(
        filepath=os.path.join(checkpoints_path, date+".hdf5"),
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        save_freq="epoch",
        monitor="val_loss"
        )

    # Adjust learning rate when our model hits a plateau (reduce overfitting)
    cb_lr = cb.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10,
                              verbose=1, mode='min', min_delta=0.0001, cooldown=1,
                              min_lr=0.00001)

    cb_es = cb.EarlyStopping(monitor="val_loss", patience=30)
    cb_tb = cb.TensorBoard(log_dir=tensorboard_path, histogram_freq=0, write_graph=True, write_images=True)
    cb_cl = cb.CSVLogger(os.path.join(csv_path, "logs.csv"), separator=',',
                      append=True)
    # cb_mc = cb.ModelCheckpoint(os.path.join(fold_subdir, "model.best.hdf5"),
    #                        monitor="loss", verbose=1,
    #                        save_best_only=True, mode="min")

    callbacks = [cb_cp, cb_es, cb_cl, cb_tb, cb_lr]

    sample_list = data_io.get_indiceslist()
    sample_list.sort(reverse=True)
    #random.shuffle(sample_list)
    print("not in val_list: " + str(sample_list[train_end_sample]))
    print("in val list: " + str(sample_list[validation_start_sample]))
    print("not in pred_list: " + str(sample_list[validation_end_sample]))
    print("in pred list: " + str(sample_list[predict_start_sample]))

    #model.evaluate(sample_list[train_start_sample:train_end_sample], sample_list[validation_start_sample:validation_end_sample], epochs=config.epochs, callbacks=callbacks)

    # Save the final weights
    model.dump(os.path.join(model_path, "model.hdf5"))

    print("\n---> Start Prediction", file=log_file)
    # Predict the segmentation of 20 samples
    model.predict(sample_list[predict_start_sample:predict_end_sample])

    overall_seg = None
    overall_pred = None
    counter = 0
    print("\n---> Create Statistics\n\n", file=log_file)
    classes_strings = [a_tuple[1] for a_tuple in config.classes]
    labels = range(0, num_of_classes)
    f1_scores = []
    IoU_scores = []
    AUC_scores = []
    MCC_scores = []
    ACC_scores = []
    sample_names = []
    ground_truth_check = 0

    for i in range(predict_start_sample, predict_end_sample):

        sample = data_io.sample_loader(sample_list[i], load_seg=True, load_pred=True)

        seg = np.squeeze(sample.seg_data, axis=-1)
        seg_img = Image.fromarray(seg)

        pred = np.squeeze(sample.pred_data, axis=-1)
        pred_img = Image.fromarray(pred)

        im = Image.fromarray(sample.img_data)
        counter += 1


        if ground_truth_check < 20:
            custom_visualizer.create_superclass_ground_truth(sample.img_data, sample.seg_data, ground_truth_path, str(sample.index))
            ground_truth_check += 1

        seg = seg.flatten()
        pred = pred.flatten()

        sample_names.append(str(sample.index))

        if num_of_classes > 1:
            tmp_dice = miseval.evaluate(seg, pred, metric="DSC", multi_class=True, n_classes=num_of_classes)
            tmp_IoU = miseval.evaluate(seg, pred, metric="IoU", multi_class=True, n_classes=num_of_classes)
            tmp_AUC = miseval.evaluate(seg, pred, metric="AUC", multi_class=True, n_classes=num_of_classes)
            tmp_MCC = miseval.evaluate(seg, pred, metric="MCC", multi_class=True, n_classes=num_of_classes)
            tmp_ACC = miseval.evaluate(seg, pred, metric="ACC", multi_class=True, n_classes=num_of_classes)
        else:
            tmp_dice = miseval.evaluate(seg, pred, metric="DSC", multi_class=False)
            tmp_IoU = miseval.evaluate(seg, pred, metric="IoU", multi_class=False)
            tmp_AUC = miseval.evaluate(seg, pred, metric="AUC", multi_class=False)
            tmp_MCC = miseval.evaluate(seg, pred, metric="MCC", multi_class=False)
            tmp_ACC = miseval.evaluate(seg, pred, metric="ACC", multi_class=False)

        # tmp_dice = np.append(tmp_dice,np.mean(tmp_dice))

        for x in range(0, num_of_classes):
            if x not in seg:
                tmp_dice[x] = -1
                tmp_IoU[x] = -1
                tmp_AUC[x] = -1
                tmp_MCC[x] = -1
                tmp_ACC[x] = -1

        f1_scores.append(tmp_dice)
        IoU_scores.append(tmp_IoU)
        AUC_scores.append(tmp_AUC)
        MCC_scores.append(tmp_MCC)
        ACC_scores.append(tmp_ACC)

        if i in random.sample(range(predict_start_sample, predict_end_sample), 10):
            tmp_save_path = os.path.join(results_path, "_sample_" + str(i))
            os.mkdir(tmp_save_path, mode)

            seg_path = os.path.join(tmp_save_path, "seg.png")
            pred_path = os.path.join(tmp_save_path, "pred.png")
            image_path = os.path.join(tmp_save_path, "image.png")
            seg_img.save(seg_path)
            pred_img.save(pred_path)
            im.save(image_path)

            for z, j in enumerate(config.classes):
                custom_visualizer.overlay_seg(image_path, seg_path, pred_path, z, j[1], tmp_save_path)
            tmp_score_log = open(os.path.join(tmp_save_path, "score_log.txt"), "w")
            print("\n---> Score File: "+str(i)+"\n---> Name: "+sample.index, file=tmp_score_log)
            print(*classes_strings, sep=" | ", file=tmp_score_log)
            print("----DICE----", file=tmp_score_log)
            print(*tmp_dice, sep=" | ", file=tmp_score_log)
            print("----IoU----", file=tmp_score_log)
            print(*tmp_IoU, sep=" | ", file=tmp_score_log)
            print("----AUC----", file=tmp_score_log)
            print(*tmp_AUC, sep=" | ", file=tmp_score_log)
            print("----MCC----", file=tmp_score_log)
            print(*tmp_MCC, sep=" | ", file=tmp_score_log)
            print("----ACC----", file=tmp_score_log)
            print(*tmp_ACC, sep=" | ", file=tmp_score_log)
            print("-1 mean that the segmentation dose not have the class. METRIC was not calculated.", file=tmp_score_log)
            #print(classification_report(seg, pred, labels=labels, target_names=classes_strings, zero_division=1), file=tmp_score_log)
            os.remove(seg_path)
            os.remove(pred_path)
            os.remove(image_path)
            tmp_score_log.close()
            # f1_scores.append(np.append(f1_score(seg, pred, average=None, labels=range(0,num_of_classes-1), zero_division=1), f1_score(seg, pred, average="weighted", zero_division=1)))

    tmp_list = []
    for sample_index, score in enumerate(f1_scores):
        for x, y in enumerate(score):
            if score[x] != -1:
                tmp_list.append([sample_names[sample_index], classes_strings[x], score[x], IoU_scores[sample_index][x], AUC_scores[sample_index][x], MCC_scores[sample_index][x], ACC_scores[sample_index][x]])
    df = pd.DataFrame(tmp_list, columns=["sample", "classes", "F1-Score", "IoU", "AUC", "MCC", "ACC"])

    # Boxplot with Seaborn

    sns.set_style('whitegrid')
    ax = sns.boxplot(x="classes", y='F1-Score', data=df, notch=True)
    ax = sns.stripplot(x="classes", y='F1-Score', data=df, linewidth=2)
    ax.set_title(config.info)
    fig = ax.get_figure()
    fig.savefig(os.path.join(boxplot_path, "boxplot_F1.png"))
    fig.clf()

    sns.set_style('whitegrid')
    ax = sns.boxplot(x="classes", y='IoU', data=df, notch=True)
    ax = sns.stripplot(x="classes", y='IoU', data=df, linewidth=2)
    ax.set_title(config.info)
    fig = ax.get_figure()
    fig.savefig(os.path.join(boxplot_path, "boxplot_IoU.png"))
    fig.clf()

    sns.set_style('whitegrid')
    ax = sns.boxplot(x="classes", y='AUC', data=df, notch=True)
    ax = sns.stripplot(x="classes", y='AUC', data=df, linewidth=2)
    ax.set_title(config.info)
    fig = ax.get_figure()
    fig.savefig(os.path.join(boxplot_path, "boxplot_AUC.png"))
    fig.clf()

    sns.set_style('whitegrid')
    ax = sns.boxplot(x="classes", y='MCC', data=df, notch=True)
    ax = sns.stripplot(x="classes", y='MCC', data=df, linewidth=2)
    ax.set_title(config.info)
    fig = ax.get_figure()
    fig.savefig(os.path.join(boxplot_path, "boxplot_MCC.png"))
    fig.clf()

    sns.set_style('whitegrid')
    ax = sns.boxplot(x="classes", y='ACC', data=df, notch=True)
    ax = sns.stripplot(x="classes", y='ACC', data=df, linewidth=2)
    ax.set_title(config.info)
    fig = ax.get_figure()
    fig.savefig(os.path.join(boxplot_path, "boxplot_ACC.png"))
    fig.clf()

    df2 = df.append(pd.Series(["","","","","","",""], index=df.columns), ignore_index=True)
    df2 = df2.append(pd.Series(["overall-mean", "all_classes", df["F1-Score"].mean(), df["IoU"].mean(), df["AUC"].mean(), df["MCC"].mean(), df["ACC"].mean()], index=df.columns), ignore_index=True)
    df2 = df2.append(pd.Series(["","","","","","",""], index=df.columns), ignore_index=True)
    df2 = df2.append(pd.Series(["medians", "table", "F1-Score", "IoU", "AUC", "MCC", "ACC"], index=df.columns), ignore_index=True)

    medians = []
    medians.append("")
    medians.append("")
    medians.append(df.groupby(['classes'])['F1-Score'].median())
    medians.append(df.groupby(['classes'])['IoU'].median())
    medians.append(df.groupby(['classes'])['AUC'].median())
    medians.append(df.groupby(['classes'])['MCC'].median())
    medians.append(df.groupby(['classes'])['ACC'].median())
    df2 = df2.append(pd.Series(medians, index=df.columns), ignore_index=True)

    print(np.count_nonzero(overall_seg==0))
    print(np.count_nonzero(overall_pred==0))
    print("\n---> Overall Score for "+str(counter)+" Sampels from the test_dataset: \n", file=log_file)
    print("---> F1: "+str(df["F1-Score"].mean()), file=log_file)
    print("---> IoU: "+str(df["IoU"].mean()), file=log_file)
    print("---> AUC: "+str(df["AUC"].mean()), file=log_file)
    print("---> MCC: "+str(df["MCC"].mean()), file=log_file)
    print("---> ACC: "+str(df["ACC"].mean()), file=log_file)

    print("\n\n---->"+config.info, file=overview_file)
    print("---> F1: "+str(df["F1-Score"].mean()), file=overview_file)
    print("---> IoU: "+str(df["IoU"].mean()), file=overview_file)
    print("---> AUC: "+str(df["AUC"].mean()), file=overview_file)
    print("---> MCC: "+str(df["MCC"].mean()), file=overview_file)
    print("---> ACC: "+str(df["ACC"].mean()), file=overview_file)

    log_file.close()
    df2.to_csv(os.path.join(results_path, "all_scores.csv"), sep='\t', encoding='utf-8', index=False)

# run configurations
date = datetime.now().strftime("%d-%m-%y-%H-%M-%S")
# Define folder structure
mode = 0o666
result_folder_path = os.path.join(".", "results")
if not os.path.exists(result_folder_path):
    os.mkdir(os.path.join(".", "results"), mode)

results_folder_path = os.path.join(result_folder_path, date)
if not os.path.exists(results_folder_path):
    os.mkdir(results_folder_path, mode)

# start tensorboard
print("\n---> Start tensorboard")
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', results_folder_path, "--host", "0.0.0.0", "--port", "6006"])
url = tb.launch()
print(f"Tensorflow listening intern on {url}")

i = 0
for config in create_configurations():
    i += 1
    print("\n---> Start Configuration: "+str(i)+": "+config.info)

    # Select only one GPU
    # check with nvidia-smi what GPU ID you need
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    log_file = open(os.path.join(results_folder_path, "overview_process.txt"), "a")
    agent(config, log_file, date, results_folder_path)
    log_file.close()