%%
%Caricamento e visualizzazione delle immagini del dataset

pathFede = 'C:\Users\fedus\Desktop\Magistrale\Primo Anno\Secondo Semestre\CV\Lab\Progetto\Data\fashion-product-images-reduced';
pathNick = '/home/nikola/UniCa/Magistrale/[1 anno - II]CV - Puglisi/Progetto';
path=pathNick;

%%
%Creazione del Datastore
imds = imageDatastore(path,'IncludeSubfolders',true,'LabelSource','foldernames');
[imdsTrain,imdsTest] = splitEachLabel(imds,0.7,'randomized');

%%
%numTrainImages = numel(imdsTrain.Labels);
%idx = randperm(numTrainImages,16);
%figure
%for i = 1:16
%    subplot(4,4,i)
%    I = readimage(imdsTrain,idx(i));
%    imshow(I)
%end

%%
%Caricamente ResNet-18
net = resnet18;
inputSize = net.Layers(1).InputSize;
analyzeNetwork(net);

%%
%Resize delle immagini
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);

%%
%Creazione pool5 (estrazione features)
layer = 'pool5';
featuresTrain = activations(net,augimdsTrain,layer,'OutputAs','rows');
featuresTest = activations(net,augimdsTest,layer,'OutputAs','rows');

whos featuresTrain

%%
%Estrazione labels
YTrain = imdsTrain.Labels;
YTest = imdsTest.Labels;

%Fit Image Classifier
classifier = fitcecoc(featuresTrain,YTrain);