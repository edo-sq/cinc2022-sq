function [dlnet] = sq_cnn(train_x)

    layers = [
        imageInputLayer([size(train_x, 1) size(train_x, 2) size(train_x, 3)], 'Name', 'input', 'Normalization', 'None')
        
        convolution2dLayer([7 9], 4, 'Name', 'conv1', 'Stride', [1 4])
        batchNormalizationLayer('Name', 'bn1')
        reluLayer('Name', 'relu1')
        
        dropoutLayer(0.5, 'Name', 'drop3')
    
        convolution2dLayer([5 7], 16, 'Name', 'conv3', 'Stride', [1 2])
        batchNormalizationLayer('Name', 'bn2')
        reluLayer('Name', 'relu3')
        
        maxPooling2dLayer([3 3], 'Name', 'pool1', 'Stride', [2 2])
        dropoutLayer(0.3, 'Name', 'drop1')
        
        convolution2dLayer([3 5], 32, 'Name', 'conv4', 'Stride', [1 1])
        batchNormalizationLayer('Name', 'bn3')
        reluLayer('Name', 'relu4')
        
        convolution2dLayer([3 1], 64, 'Name', 'conv5', 'Stride', [1 1])
        batchNormalizationLayer('Name', 'bn4')
        reluLayer('Name', 'relu5')
        
        maxPooling2dLayer([3 1], 'Name', 'pool2', 'Stride', [2 1])
        
        convolution2dLayer([3 1], 128, 'Name', 'conv6', 'Stride', [1 1])
        batchNormalizationLayer('Name', 'bn5')
        reluLayer('Name', 'relu6') 
        
        convolution2dLayer([3 1], 256, 'Name', 'conv7', 'Stride', [1 1])
        batchNormalizationLayer('Name', 'bn6')
        reluLayer('Name', 'relu7')
        
        maxPooling2dLayer([3 1], 'Name', 'pool3', 'Stride', [2 1])
    
        dropoutLayer(0.5, 'Name', 'drop3')
    
        fullyConnectedLayer(256, 'Name', 'fc_preprelast')
        
        fullyConnectedLayer(64, 'Name', 'fc_prelast')
        reluLayer('Name', 'relu8')
        
        fullyConnectedLayer(2, 'Name', 'fc_last')
        softmaxLayer('Name', 'softmax')
    ];
%     net = layerGraph(layers);
    dlnet = dlnetwork(layers);

end