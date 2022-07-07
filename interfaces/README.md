Interfaces refer to notebook and python files for performing tasks.



# mixture_net (THz) interfaces

    ├── handling_mixture_THz_data.ipynb (also acts as data generator for training and testing)
    ├── mixture_THz_CV.ipynb (Cross-validation on THz mixture data)
    ├── mixture_THz_train_test_split.ipynb (perform training and testing by generating or loading mixture data)
    └── mixture_THz_result_postprocessor.ipynb (postprocess the model outputs to get metrics)

# CNN interfaces

    ├── handling_IR_CNN_data.ipynb (Demonstrates the conversion from spectra to img)
    ├── IR_CNN_train_test_split.ipynb (Try out CNN architechtures on simulated IR data)
    ├── IR_CNN1D_train_test_split.ipynb (1D CNN on simulated IR data)
    ├── THz_CNN1D_train_test_split.ipynb (1D CNN on simulated THz data)
    └── THz_CNN_train_test_split.ipynb (Try out CNN architechtures on simulated IR data)



# IR interfaces

    ├── handling_IR_data.ipynb
    ├── IR_Classifier_Only_With_Sim_Valid.ipynb
    ├── IR_exp_clf_gui.ipynb ( A notebook GUI for identfying pure IR spectra)
    ├── IR_SVM_Constant_Features.ipynb (Effects of resolution at constant number of features ar analyzed here)
    ├── IR_SVM_Constant_Resolution.ipynb (Effects of number of features at constant resolution analyzed here)
    ├── IR_SVM_CV.ipynb (cross-validation of the SVM classifier)
    ├── IR_SVM_Hyperparameter.ipynb (Effect of hyperparameter C is explored))
    ├── IR_SVM_train_test_split.ipynb (70%-30% training-testing split for SVM classifier training))
    ├── kmeans_IR.ipynb (k-means analysis on IR data)
    ├── PCA_IR.ipynb (PCA analysis on IR data)
    └── tsne_IR.ipynb (tSNE analysis of IR spectra))

# THz interfaces

    ├── handling_datasets.ipynb (how to handle THz datasets)
    ├── KFOLDFIGURES (stores the images of confusion matrices)
    ├── kmeans_analysis.ipynb (k-means clustering analysis of THz spectra)
    ├── THz_cross_validation.ipynb (Training classifiers and cross-validation)
    ├── THz_experimental_clf.ipynb (Identifying experiments using trained classifiers)
    ├── THz_train_test_simulator.py (Faster execution of 70%-30% training-testing of classifiers)
    ├── THz_train_test_split.ipynb (Detailed notebook of 70%-30% train-test split and experimental identification)
    └── tsne_analysis.ipynb (tSNE analysis of THz spectra)

