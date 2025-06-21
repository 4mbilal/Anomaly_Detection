clear all
close all
clc
gpuDevice(1);
rng_seed = randi(1000);
Options = getOptions;
Options = getEmbeddingsModel(Options);
% analyzeNetwork(Options.net);
% return

auc_array = [];
TestScores = [];
doTrain = true;
% doTrain = false;

for i=1:numel(Options.classes)
% for i=13:13
    [tdsTrain, tdsTest, tdsMasks] = prepareData(fullfile(Options.dataDir, Options.classes(i)),Options);
    figure
    if(doTrain)
        [XTrainEmbeddings, TrainLabels] = getFeatures(tdsTrain,Options);
        for k=1:7
            [XTrainEmbeddings_, TrainLabels] = getFeatures(tdsTrain,Options);
            XTrainEmbeddings.f1 = cat(4,XTrainEmbeddings.f1,XTrainEmbeddings_.f1);
            XTrainEmbeddings.f2 = cat(4,XTrainEmbeddings.f2,XTrainEmbeddings_.f2);
            XTrainEmbeddings.f3 = cat(4,XTrainEmbeddings.f3,XTrainEmbeddings_.f3);
        end        
    
        [PCAdata1,PCAdata2,PCAdata3] = getFeaturesPCA(XTrainEmbeddings,Options);
        % save(strcat(Options.classes(i),"_PCAdata.mat"),"PCAdata1","PCAdata2","PCAdata3");
    end
    % keyboard
    clear XTrainEmbeddings %Save memory by deleting these embeddings.
    % load(strcat(Options.classes(i),"_PCAdata.mat"),"PCAdata1","PCAdata2","PCAdata3");
    [XTestEmbeddings, Testlabels] = getFeatures(tdsTest,Options);
    [auc_i,auc_p,ts] = detectAnomalies(XTestEmbeddings,Testlabels,Options,PCAdata1,PCAdata2,PCAdata3,tdsMasks);
    auc_array = [auc_array; [auc_i,auc_p]]
    TestScores{i} = ts;
    % close all
end
m_auc_i_p = mean(auc_array,1)


function XTrainEmbeddings = concat_embeddings(E1,E2,E3,E4)
XTrainEmbeddings.f1 = cat(4,E1.f1,E2.f1,E3.f1,E4.f1);
XTrainEmbeddings.f2 = cat(4,E1.f2,E2.f2,E3.f2,E4.f2);
XTrainEmbeddings.f3 = cat(4,E1.f3,E2.f3,E3.f3,E4.f3);

end

function Options = getOptions
Options.resizeImageSize = [256 256]*1;
Options.targetImageSize = [224 224]*1;
% Options.targetImageSize = [256 256]*1;

% Options.resizeImageSize = [256 256]*1;
% Options.targetImageSize = [227 227]*1;

% Options.resizeImageSize = [512 512]*1;
% Options.targetImageSize = [416 416]*1;

Options.dataDir = 'D:\RnD\Frameworks\Datasets\anomaly\mvtec_anomaly_detection\';
% Options.dataDir = 'C:\anomaly\datasets\mvtec_anomaly_detection\';
% Options.dataDir = 'D:\Datasets\mvtec_anomaly_detection\';
% Options.dataDir = 'D:\RnD\Frameworks\Datasets\anomaly\VisA\matlab\';

Options.classes = ["carpet" "grid" "leather" "tile" "wood" "bottle" "cable" "capsule" "hazelnut" "metal_nut" "pill" "screw" "toothbrush" "transistor" "zipper"];
% Options.classes = ["grid"];

% Options.classes = ["candle" "capsules" "cashew" "chewinggum" "fryum" "macaroni1" "macaroni2" "pcb1" "pcb2" "pcb3" "pcb4" "pipe_fryum"];
% Options.classes = ["chewinggum"];
end