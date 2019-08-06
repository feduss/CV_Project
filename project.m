
%Individuazione della path delle immagini da analizzare
clc;
clearvars;

tic;
pathFede = 'C:\Users\fedus\Desktop\Magistrale\Primo Anno\Secondo Semestre\CV\Lab\Progetto\Data\fashion-product-images-dataset\fashion-dataset\pure_images';
pathNick = '/home/nikola/UniCa/Magistrale/[1 anno - II]CV - Puglisi/Progetto';
path=pathFede;


%Creazione del Datastore
%Le label sono prese direttamente dal nome della cartella che contiene
%l'immagine. Le immagini sono state spostate nelle rispettive cartelle
%mediante una funzione python creata da noi
imds = imageDatastore(path,'IncludeSubfolders',true,'LabelSource','foldernames');
%Divido le labello delle immagini in set Train e Test
[imdsTrain,imdsTest] = splitEachLabel(imds,0.7,'randomized');


%Caricamente ResNet-18
%net = resnet18;
net = alexnet;
inputSize = net.Layers(1).InputSize;
analyzeNetwork(net);


%Resize delle immagini per rispettare i requisiti del network
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);


%Creazione pool5 (estrazione features)
%layer = 'pool5'; %resnet18
layer = 'prob'; %alexnet
%Estrazione delle feature per il set Train e Test
featuresTrain = activations(net,augimdsTrain,layer,'OutputAs','rows');
featuresTest = activations(net,augimdsTest,layer,'OutputAs','rows');

whos featuresTrain


%Estrazione labels
YTrain = imdsTrain.Labels;
YTest = imdsTest.Labels;

%1)Classificatore fitcecoc (SVM multiscala) con successiva predizione
classifier = fitcecoc(featuresTrain,YTrain);

YPred = predict(classifier,featuresTest);

%Calcolo della precisione media
accuracy_svm_multi = mean(YPred == YTest)
tempo_estrazione_feature = toc

%
tic;
list_label1 = YTrain;
list_label2 = YTrain;
list_label3 = YTrain;

list_label1(list_label1 ~= 'Men-Apparel-Topwear-Tshirts') = 'false';
list_label2(list_label2 ~= 'Men-Apparel-Topwear-Shirts') = 'false';
list_label3(list_label3 ~= 'Men-Footwear-Shoes-Casual Shoes') = 'false';

list_label1_test = YTest;
list_label2_test = YTest;
list_label3_test = YTest;

list_label1_test(list_label1_test ~= 'Men-Apparel-Topwear-Tshirts') = 'false';
list_label2_test(list_label2_test ~= 'Men-Apparel-Topwear-Shirts') = 'false';
list_label3_test(list_label3_test ~= 'Men-Footwear-Shoes-Casual Shoes') = 'false';

tempo_creazione_label = toc;

%SVM binaria e Predizione
tic;
SVMModel1 = fitcsvm(featuresTrain,list_label1,'KernelFunction','rbf','Standardize',true,'ClassNames',{'false','Men-Apparel-Topwear-Tshirts'});
SVMModel2 = fitcsvm(featuresTrain,list_label2,'KernelFunction','rbf','Standardize',true,'ClassNames',{'false','Men-Apparel-Topwear-Shirts'});
SVMModel3 = fitcsvm(featuresTrain,list_label3,'KernelFunction','rbf','Standardize',true,'ClassNames',{'false','Men-Footwear-Shoes-Casual Shoes'});

tempo_creazione_svm = toc

tic;
[label1_svm,score1_svm] = predict(SVMModel1,featuresTest);
[label2_svm,score2_svm] = predict(SVMModel2,featuresTest);
[label3_svm,score3_svm] = predict(SVMModel3,featuresTest);

tempo_predict_svm = toc
%Accuracy svm bin

accuracy_svm1 = mean (label1_svm == list_label1_test);
accuracy_svm2 = mean (label2_svm == list_label2_test);
accuracy_svm3 = mean (label3_svm == list_label3_test);

mean_accuracy_svm = (accuracy_svm1 + accuracy_svm2 + accuracy_svm3)/3


%Knn e predict

Mdl = fitcknn(featuresTrain,YTrain,'NumNeighbors',5,'Standardize',1);
[label_knn,score_knn,cost_knn] = predict(Mdl,featuresTest);
accuracy_knn = mean (label_knn == YTest);

tic;
Mdl1 = fitcknn(featuresTrain,list_label1,'NumNeighbors',5,'Standardize',1);
Mdl2 = fitcknn(featuresTrain,list_label2,'NumNeighbors',5,'Standardize',1);
Mdl3 = fitcknn(featuresTrain,list_label3,'NumNeighbors',5,'Standardize',1);
tempo_creazione_knn = toc


tic;
[label1_knn,score1_knn,cost1_knn] = predict(Mdl1,featuresTest);
[label2_knn,score2_knn,cost2_knn] = predict(Mdl2,featuresTest);
[label3_knn,score3v,cost3_knn] = predict(Mdl3,featuresTest);

tempo_predict_knn = toc


%Accuracy k-nn

accuracy_knn1 = mean (label1_knn == list_label1_test);
accuracy_knn2 = mean (label2_knn == list_label2_test);
accuracy_knn3 = mean (label3_knn == list_label3_test);

mean_accuracy_svm = (accuracy_knn1 + accuracy_knn2 + accuracy_knn3)/3

[sym('svm_multi'), sym('svm_bin1'), sym('svm_bin2'), sym('svm_bin3'), sym('knn1'), sym('knn2'), sym('knn3'),
    accuracy_svm_multi, accuracy_svm1, accuracy_svm2, accuracy_svm3, accuracy_knn1, accuracy_knn2, accuracy_knn3]
