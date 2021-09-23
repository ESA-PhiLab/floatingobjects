import sys
import os
from data import FloatingSeaObjectDataset
from visualization import plot_batch, calculate_fdi, ndvi_transform, s2_to_RGB, plot_curves
from train import predict_images
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from visualization import calculate_fdi, ndvi_transform
import torch
import rasterio
from tqdm.auto import tqdm as tq
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix
from model import UNet, get_model
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

torch.multiprocessing.set_sharing_strategy('file_system')

N_PIXELS_FOR_EACH_CLASS_FROM_IMAGE = 5
LABELS = ["no floating", "floating"]

device = "cuda" if torch.cuda.is_available() else "cpu"


def sample_N_random(data, N):
    idxs = np.random.choice(np.arange(len(data)), min(len(data), N), replace=False)
    return data[idxs]


def aggregate_images(x, y):
    """
    aggregates images to pixel datasets
    x (image): D x H x W -> N x D where N randomly sampled pixels within image
    y (label): H x W -> N
    """
    N = N_PIXELS_FOR_EACH_CLASS_FROM_IMAGE

    floating_objects = x[:, y.astype(bool)].T  # 1: floating
    not_floating_objects = x[:, ~y.astype(bool)].T  # 0: no floating

    # use less if len(floating_objects) < N
    N = min(N, len(floating_objects))

    x_floating_objects = sample_N_random(floating_objects, N)
    y_floating_objects = np.ones(x_floating_objects.shape[0], dtype=int)

    x_not_floating_objects = sample_N_random(not_floating_objects, N)
    y_not_floating_objects = np.zeros(x_not_floating_objects.shape[0], dtype=int)

    x = np.vstack([x_floating_objects, x_not_floating_objects])
    y = np.vstack([y_floating_objects, y_not_floating_objects]).reshape(-1)
    return x, y


def feature_extraction_transform(x, y):
    x, y = aggregate_images(x, y)

    # transforms require band dimension first:
    # so N x D -transpose> D x N -ndvi/fdi> N
    ndvi = ndvi_transform(x.T)
    fdi = calculate_fdi(x.T)

    # N x 2
    return np.vstack([fdi, ndvi]).T, y


def s2_to_ndvifdi(x):
    ndvi = ndvi_transform(x)
    fdi = calculate_fdi(x)
    return np.stack([fdi, ndvi])


def draw_N_datapoints(dataset, N):
    idxs = np.random.randint(len(dataset), size=N)

    x = []
    y = []
    for idx in idxs:
        x_,y_,fid_ = dataset[idx]
        x.append(x_)
        y.append(y_)

    return np.vstack(x), np.hstack(y)


# Plot the confusion matrix
def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


def metrics(cm, label_values=LABELS):
    print("Confusion matrix:")
    print(cm)
    print("---")

    # Compute global accuracy
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    print("{} pixels processed".format(total))
    print("Total accuracy: {}%".format(accuracy))
    print("---")

    # Compute F1 score
    F1Score = np.zeros(len(label_values))
    for i in range(len(label_values)):
        try:
            F1Score[i] = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except:
            # Ignore exception if there is no element in class i for test set
            pass
    print("F1-Score:")
    for l_id, score in enumerate(F1Score):
        print("{}: {}".format(label_values[l_id], score))
    print("---")

    # Compute kappa coefficient
    total = np.sum(cm)
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
    kappa = (pa - pe) / (1 - pe)
    print("Kappa: " + str(kappa))
    print("---")
    #return accuracy, F1Score, kappa

#############################################################################################################
#data_path = "/home/jmifdal/data/floatingobjects"
data_path = "/home/raquel/floatingobjects/data"
#image_size = (128, 128)
image_size = 128
fold_set = "val"
seed = 1
net = 'manet'

get_preds = False # get predictions from DL models on fold_set
sota = False # get predictions from sota on fold_set

check_metrics = False # check performance metrics from predictions
threshold = 0.03
N_pixels = 20000

if get_preds:
    # dataset for training with the feature extraction transform
    dataset = FloatingSeaObjectDataset(data_path, fold="train", seed=seed, transform=feature_extraction_transform,
                                    output_size=image_size, hard_negative_mining=False, use_l2a_probability=1)

    # dataset = FloatingSeaObjectDataset(data_path, fold="train", seed=seed, transform=feature_extraction_transform,
    #                                   output_size=image_size)

    # dataset for training with no transform
    imagedataset = FloatingSeaObjectDataset(data_path, fold="val", seed=seed, transform=None,
                                            output_size=image_size, hard_negative_mining=False, use_l2a_probability=1)

    # dataset for validation with the feature training transform
    test_dataset = FloatingSeaObjectDataset(data_path, fold="val", seed=seed, transform=feature_extraction_transform,
                                            output_size=image_size, hard_negative_mining=False, use_l2a_probability=1)

    # dataset for validation with no feature extraction
    testimagedataset = FloatingSeaObjectDataset(data_path, fold=fold_set, seed=seed, transform=None, output_size=image_size)


    if sota:
        #### Support Vector Classification ####
        from sklearn import svm
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report
        x,y = draw_N_datapoints(dataset, N=1000)
        clf_svm = svm.SVC(gamma=1000, C=30)
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
        clf_svm.fit(X_train, y_train)
        y_pred = clf_svm.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=["water","floating objects"]))


        #### Naive Bayes Classifier ####
        from sklearn.naive_bayes import GaussianNB
        #x,y = draw_N_datapoints(dataset, N=1000)
        clf_nb = GaussianNB()
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
        clf_nb.fit(X_train, y_train)
        y_pred = clf_nb.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=["water","floating objects"]))


        #### Random Forest Classifier ####
        from sklearn.ensemble import RandomForestClassifier
        #x,y = draw_N_datapoints(dataset, N=1000)
        clf_rf = RandomForestClassifier(n_estimators=1000, max_depth=2)
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
        clf_rf.fit(X_train, y_train)
        y_pred = clf_rf.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=["water","floating objects"]))


        #### Hist-based Gradient Boosting Classifier ####
        from sklearn.experimental import enable_hist_gradient_boosting  # noqa
        from sklearn.ensemble import HistGradientBoostingClassifier
        #x,y = draw_N_datapoints(dataset, N=1000)
        clf_hgb = HistGradientBoostingClassifier()
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
        clf_hgb.fit(X_train, y_train)
        y_pred = clf_hgb.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=["water","floating objects"]))

    ######## Trained model
    # path to the model
    #model_path=os.environ['HOME'] + '/remote/floatingobjects/models/model_24_12_2020.pth.tar'
    #model_path=os.environ['HOME'] + '/remote/floatingobjects/models/model_19_01_2021.pth.tar'
    #model_path=os.environ['HOME'] + '/remote/floatingobjects/models/model_ratio10_22_01_2021.pth.tar'
    model_path = f'models/{net}-cross-val-2fold/model_{seed}.pth.tar'
    print(model_path)

    #model = UNet(n_channels=12, n_classes=1, bilinear=False).to(device)
    model = get_model(net, inchannels=12).to(device)
    model.load_state_dict(torch.load(model_path)["model_state_dict"])

    #### Test
    test_loader = DataLoader(testimagedataset, batch_size=1, shuffle=False, num_workers=2)

    #confusion matrix
    conf_mat = np.zeros((len(LABELS), len(LABELS)))

    outputs = []
    y_trues = []

    # SVM, RF, NB and HGB
    y_trues_sota = []
    y_preds_svm = []
    y_preds_rf = []
    y_preds_nb = []
    y_preds_hgb = []

    for idx, (image, y_true, _) in tq(enumerate(test_loader), total=len(test_loader)):
        if sota:
            # apply SVM, RF, NB and HGB methods
            features = s2_to_ndvifdi(image.squeeze(0).numpy())
            y_trues_sota.append(y_true.squeeze(0))
            
            # svm
            y_pred_svm = clf_svm.predict(features.reshape(2, -1).T).reshape(128, 128)
            y_preds_svm.append(y_pred_svm)

            # rf
            y_pred_rf = clf_rf.predict(features.reshape(2, -1).T).reshape(128, 128)
            y_preds_rf.append(y_pred_rf)

            # nb
            y_pred_nb = clf_nb.predict(features.reshape(2, -1).T).reshape(128, 128)
            y_preds_nb.append(y_pred_nb)
            
            # hgb
            y_pred_hgb = clf_hgb.predict(features.reshape(2, -1).T).reshape(128, 128)
            y_preds_hgb.append(y_pred_hgb)

        # unet
        image = image.to(device, dtype=torch.float)
        y_true = y_true.to(device)
        y_trues.append(y_true)

        with torch.no_grad():
            # forward pass: compute predicted outputs by passing inputs to the model
            logits = model.forward(image)
            output = torch.sigmoid(logits)
            outputs.append(output)


    print("hallo!")
    #metrics_dir = f"metrics/unet-posweight10-lr001-aug1/{fold_set}/"
    metrics_dir = f"metrics/{net}-cross-val-2fold/model_{seed}/{fold_set}/"
    os.makedirs(metrics_dir, exist_ok=True)
    torch.save(outputs, metrics_dir + 'outputs.pt')
    torch.save(y_trues, metrics_dir + 'y_trues.pt')

    if sota:
        metrics_dir = f"metrics/{net}-cross-val-2fold/sota_{seed}/{fold_set}/"
        os.makedirs(metrics_dir, exist_ok=True)
        torch.save(y_trues_sota, metrics_dir + 'y_trues_sota.pt')
        torch.save(y_preds_svm, metrics_dir + 'y_preds_svm.pt')
        torch.save(y_preds_rf, metrics_dir + 'y_preds_rf.pt')
        torch.save(y_preds_nb, metrics_dir + 'y_preds_nb.pt')
        torch.save(y_preds_hgb, metrics_dir + 'y_preds_hgb.pt')

######################################################################################
if check_metrics:
    # Check metrics
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    prec = dict()
    recall = dict()

    metrics_dir = f"metrics/unet-cross-val-2fold/sota_{seed}/{fold_set}/"
    print("Model:", os.path.normpath(metrics_dir).split('/')[-2:])
    y_trues_sota = torch.load(metrics_dir + 'y_trues_sota.pt')
    y_preds_svm = torch.load(metrics_dir + 'y_preds_svm.pt')
    y_preds_rf = torch.load(metrics_dir + 'y_preds_rf.pt')
    y_preds_nb = torch.load(metrics_dir + 'y_preds_nb.pt')
    y_preds_hgb = torch.load(metrics_dir + 'y_preds_hgb.pt')

    ### SVM
    print("Model: SVM")
    y_pred_svm_flat=np.vstack(y_preds_svm).reshape(-1)
    y_true_svm_flat=np.vstack(y_trues_sota).reshape(-1)
    idx_floating, = np.where(y_true_svm_flat.astype(bool))
    idx_water, = np.where(~y_true_svm_flat.astype(bool))
    idx_floating_choice = np.random.choice(idx_floating,N_pixels)
    idx_water_choice = np.random.choice(idx_water,N_pixels)
    idx = np.hstack([idx_floating_choice,idx_water_choice])
    #y_pred_svm_flat = y_pred_svm_flat[idx] > threshold
    conf_mat = confusion_matrix(y_true_svm_flat[idx], y_pred_svm_flat[idx] > threshold)
    #print(conf_mat)
    metrics(conf_mat,LABELS)
    fpr['svm'], tpr['svm'], _ = roc_curve(y_true_svm_flat[idx], y_pred_svm_flat[idx])
    roc_auc['svm'] = auc(fpr['svm'], tpr['svm'])
    prec['svm'], recall['svm'], _ = precision_recall_curve(y_true_svm_flat[idx], y_pred_svm_flat[idx])

    ### RF
    print("\nModel: RF")
    y_pred_rf_flat=np.vstack(y_preds_rf).reshape(-1)
    y_true_rf_flat=np.vstack(y_trues_sota).reshape(-1)
    idx_floating, = np.where(y_true_rf_flat.astype(bool))
    idx_water, = np.where(~y_true_rf_flat.astype(bool))
    idx_floating_choice = np.random.choice(idx_floating,N_pixels)
    idx_water_choice = np.random.choice(idx_water,N_pixels)
    idx = np.hstack([idx_floating_choice,idx_water_choice])
    #y_pred_rf_flat = y_pred_rf_flat[idx] > threshold
    conf_mat = confusion_matrix(y_true_rf_flat[idx], y_pred_rf_flat[idx] > threshold)
    metrics(conf_mat,LABELS)
    fpr['rf'], tpr['rf'], _ = roc_curve(y_true_rf_flat[idx], y_pred_rf_flat[idx])
    roc_auc['rf'] = auc(fpr['rf'], tpr['rf'])
    prec['rf'], recall['rf'], _ = precision_recall_curve(y_true_rf_flat[idx], y_pred_rf_flat[idx])

    ### NB
    print("\nModel: NB")
    y_pred_nb_flat=np.vstack(y_preds_nb).reshape(-1)
    y_true_nb_flat=np.vstack(y_trues_sota).reshape(-1)
    idx_floating, = np.where(y_true_nb_flat.astype(bool))
    idx_water, = np.where(~y_true_nb_flat.astype(bool))
    idx_floating_choice = np.random.choice(idx_floating,N_pixels)
    idx_water_choice = np.random.choice(idx_water,N_pixels)
    idx = np.hstack([idx_floating_choice,idx_water_choice])
    #y_pred_nb_flat = y_pred_nb_flat[idx] > threshold
    conf_mat = confusion_matrix(y_true_nb_flat[idx], y_pred_nb_flat[idx] > threshold)
    metrics(conf_mat, LABELS)
    fpr['nb'], tpr['nb'], _ = roc_curve(y_true_nb_flat[idx], y_pred_nb_flat[idx])
    roc_auc['nb'] = auc(fpr['nb'], tpr['nb'])
    prec['nb'], recall['nb'], _ = precision_recall_curve(y_true_nb_flat[idx], y_pred_nb_flat[idx])

    ### HGB
    print("\nModel: HGB")
    y_pred_hgb_flat=np.vstack(y_preds_hgb).reshape(-1)
    y_true_hgb_flat=np.vstack(y_trues_sota).reshape(-1)
    idx_floating, = np.where(y_true_hgb_flat.astype(bool))
    idx_water, = np.where(~y_true_hgb_flat.astype(bool))
    idx_floating_choice = np.random.choice(idx_floating,N_pixels)
    idx_water_choice = np.random.choice(idx_water,N_pixels)
    idx = np.hstack([idx_floating_choice,idx_water_choice])
    #y_pred_hgb_flat = y_pred_hgb_flat[idx] > threshold
    conf_mat = confusion_matrix(y_true_hgb_flat[idx], y_pred_hgb_flat[idx] > threshold)
    metrics(conf_mat, LABELS)
    fpr['hgb'], tpr['hgb'], _ = roc_curve(y_true_hgb_flat[idx], y_pred_hgb_flat[idx])
    roc_auc['hgb'] = auc(fpr['hgb'], tpr['hgb'])
    prec['hgb'], recall['hgb'], _ = precision_recall_curve(y_true_hgb_flat[idx], y_pred_hgb_flat[idx])


    ### U-NETs
    #nets = ['unet-posweight1-lr001-aug1', 'unet-posweight10-lr001-aug1', 'resnetunet-posweight1-lr001-aug1', 
    #        'resnetunetscse-posweight1-lr001-aug1', 'manet-posweight1-lr001-aug1']
    #nets = [0, 1] # number of cross-val folds
    nets = ['unet-cross-val-2fold', 'manet-cross-val-2fold']

    accs = []
    f1s_w = []
    f1s_o = []
    kappas = []

    for net in nets:
        metrics_dir = f"metrics/{net}/model_{seed}/{fold_set}/"
        print("\nModel:", os.path.normpath(metrics_dir).split('/')[-2:])

        outputs = torch.load(metrics_dir + 'outputs.pt')
        y_trues = torch.load(metrics_dir + 'y_trues.pt')

        outputs_cpu = [i.cpu() for i in outputs]
        y_trues_cpu = [i.cpu() for i in y_trues]

        y_pred_cnn_flat=np.vstack(outputs_cpu).reshape(-1)
        y_true_cnn_flat=np.vstack(y_trues_cpu).reshape(-1)
        idx_floating, = np.where(y_true_cnn_flat.astype(bool))
        idx_water, = np.where(~y_true_cnn_flat.astype(bool))
        idx_floating_choice = np.random.choice(idx_floating,N_pixels)
        idx_water_choice = np.random.choice(idx_water,N_pixels)
        idx = np.hstack([idx_floating_choice,idx_water_choice])
        #y_pred_cnn_flat = y_pred_cnn_flat[idx] > threshold
        #conf_mat = confusion_matrix(y_true_cnn_flat[idx], y_pred_cnn_flat)
        conf_mat = confusion_matrix(y_true_cnn_flat[idx], y_pred_cnn_flat[idx] > threshold)
        metrics(conf_mat, LABELS)
        #acc, f1, kappa = metrics(conf_mat, LABELS)
        #accs.append(acc)
        #f1s_w.append(f1[0])
        #f1s_o.append(f1[1])
        #kappas.append(kappa)
        name = 'U-Net' if net=='unet-cross-val-2fold' else 'MA-Net'
        fpr[name], tpr[name], _ = roc_curve(y_true_cnn_flat[idx], y_pred_cnn_flat[idx])
        roc_auc[name] = auc(fpr[name], tpr[name])
        prec[name], recall[name], _ = precision_recall_curve(y_true_cnn_flat[idx], y_pred_cnn_flat[idx])

        #print("accs:", accs, np.mean(np.asarray(accs)), np.std(np.asarray(accs)))
        #print("f1s_w:", f1s_w, np.mean(np.asarray(f1s_w)), np.std(np.asarray(f1s_w)))
        #print("f1s_o:", f1s_o, np.mean(np.asarray(f1s_o)), np.std(np.asarray(f1s_o)))
        #print("kappas:", kappas, np.mean(np.asarray(kappas)), np.std(np.asarray(kappas)))

    # plot ROC and Recall/Precision curves
    fig = plot_curves(nets, fpr, tpr, roc_auc, recall, prec)
    fig.savefig(f"metrics/plots/20210721_{fold_set}_curves_{seed}.jpg", bbox_inches='tight')