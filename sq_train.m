function [best_net] = sq_train(train_ds, val_dla, val_y, dlnet, classes)

%weights = [[0,0,2];[0,0,1];[2,1,0]];


% classes = categories(test_y);
% [h,w,c,~] = size(train_x);
% imageInputSize = [h w c];
val_yoh = onehotencode(val_y, 2)';
pidx = val_y == "Present";
aidx = val_y == "Absent";

miniBatchSize = 32;
mbq = minibatchqueue(train_ds, 'MiniBatchSize',miniBatchSize,'MiniBatchFcn', @preprocessMiniBatch,'MiniBatchFormat',{'SSCB',''});

avgGrad = [];
avgSqGrad = [];

numEpochs = 40;

iteration = 0;
start = tic;

best_val = 0;
best_net = dlnet;

best_val2 = Inf;
best_net2 = dlnet;

% Loop over epochs.
for epoch = 1:numEpochs
    % Shuffle data.
    shuffle(mbq);
    
    % Loop over mini-batches.
    while hasdata(mbq)
        iteration = iteration + 1;
        
        % Read mini-batch of data.
        [dlX, dlY] = next(mbq);
        
        % Evaluate the model gradients, state, and loss using dlfeval and the
        % modelGradients function and update the network state.
        [gradients,state,loss] = dlfeval(@modelGradients,dlnet,dlX,dlY);
        dlnet.State = state;
        
        % Update the network parameters using the SGDM optimizer.
        [dlnet,avgGrad,avgSqGrad] = adamupdate(dlnet,gradients,avgGrad,avgSqGrad,iteration);
        
        % Display the training progress.
        D = duration(0,0,toc(start),'Format','hh:mm:ss');
        
        if iteration == 1 || (mod(iteration, 10) == 0)
            yp = predict(dlnet, val_dla);

            valLoss = double(gather(extractdata(crossentropy(yp, val_yoh))));
            predictions = onehotdecode(yp, classes,1)';
            YPred = predictions;
            valBAcc = sq_bacc(val_y, YPred) * 100;
            valAAcc = mean(YPred(aidx) == val_y(aidx)) * 100;
            valPAcc = mean(YPred(pidx) == val_y(pidx)) * 100;
            disp(valPAcc);

            if (valBAcc > best_val)
                best_val = valBAcc;
                best_net = dlnet;
            end
            
        end
        
        % Update graphs.
        drawnow
    end
end

end

function [gradients,state,loss] = modelGradients(dlnet,dlX,Y)

[dlYPred,state]=forward(dlnet,dlX);

weights = [1, 1.5];
loss = crossentropy(dlYPred,Y, weights, 'WeightsFormat', 'UC');
% weights = [[0,0,2];[0,0,1];[2,1,0]];
% loss = crossentropy(dlYPred,Y);
% loss = sqv_crossentropy(dlYPred, Y, weights);
gradients = dlgradient(loss,dlnet.Learnables);

loss = double(gather(extractdata(loss)));

end

function [X,Y] = preprocessMiniBatch(XCell,YCell)

% Preprocess predictors.
X = preprocessMiniBatchPredictors(XCell);

% Extract label data from cell and concatenate.
Y = cat(2,YCell{:});

% One-hot encode labels.
Y = onehotencode(Y,1);

end

function X = preprocessMiniBatchPredictors(XCell)

% Concatenate.
X = cat(4,XCell{:});

end

function [bacc] = sq_bacc(yt, yp)
    cyt = categories(yt);
%     cyp = categories(yp);
%     inter = intersect(cyt, cyp);
%     if length(inter) ~= length(cyt) || length(inter) ~= length(cyp)
%         disp("ERROR: categories do no match. Returning NaN.");
%         res = NaN;
%         return;
%     end
    accs = zeros(length(cyt), 1);
    for i = 1:length(cyt)
        c = string(cyt{i});
        cidx = yt == c;
        accs(i) = mean(yt(cidx) == yp(cidx));
    end
    bacc = mean(accs);
end