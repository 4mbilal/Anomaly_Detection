function [PCAdata1,PCAdata2,PCAdata3] = getFeaturesPCA(Embeddings,Options)
N1 = round(0.8*size(Embeddings.f1,3));
N2 = round(0.8*size(Embeddings.f2,3));
N3 = round(0.8*size(Embeddings.f3,3));

% N1 = round(0.5*size(Embeddings.f1,3));
% N2 = round(0.5*size(Embeddings.f2,3));
% N3 = round(0.5*size(Embeddings.f3,3));


% N1 = size(Embeddings.f1,3);
% N2 = size(Embeddings.f2,3);
% N3 = size(Embeddings.f3,3);


K = Options.No_of_output_layers;
PCAdata1 = [];
PCAdata2 = [];
PCAdata3 = [];

switch K
    case 1
        Embeddings1 = gather(extractdata(Embeddings.f1));
        [PCAdata1.means,PCAdata1.xforms,PCAdata1.eigval] = PCA_helper(Embeddings1,N1);
    case 2          
        Embeddings1 = gather(extractdata(Embeddings.f1));
        [PCAdata1.means,PCAdata1.xforms,PCAdata1.eigval] = PCA_helper(Embeddings1,N1);
        Embeddings2 = gather(extractdata(Embeddings.f2));
        [PCAdata2.means,PCAdata2.xforms,PCAdata2.eigval] = PCA_helper(Embeddings2,N2);
    case 3    
        Embeddings1 = gather(extractdata(Embeddings.f1));
        [PCAdata1.means,PCAdata1.xforms,PCAdata1.eigval] = PCA_helper(Embeddings1,N1);
        Embeddings2 = gather(extractdata(Embeddings.f2));
        [PCAdata2.means,PCAdata2.xforms,PCAdata2.eigval] = PCA_helper(Embeddings2,N2);
        Embeddings3 = gather(extractdata(Embeddings.f3));
        [PCAdata3.means,PCAdata3.xforms,PCAdata3.eigval] = PCA_helper(Embeddings3,N3);
end

end

function [means,xforms,eigval] = PCA_helper(Embeddings,N)

    [H, W, C, B] = size(Embeddings);
    XTrainEmbeddings = reshape(Embeddings,[H*W C B]);
    means = mean(XTrainEmbeddings,3);
    xforms = zeros([H*W C N]);
    eigval = zeros([H*W N]);
    % keyboard

    % identityMatrix = eye(C);
    for idx = 1:H*W
        channel_data = squeeze(XTrainEmbeddings(idx,:,:))';
        
        channel_data_zero_mean = channel_data-means(idx,:);
        S = (channel_data_zero_mean'*channel_data_zero_mean)/(length(channel_data_zero_mean)-1);
    
    % try
        [eigvec,eigval_] = eig(S);
    % catch
    %     % keyboard
    %     [eigvec,eigval_] = eig(S+eps);
    %     disp('eig function failed')
    % end
        % Xform = fliplr(eigvec);
        % eigval = flipud(diag(eigval));
        eigval_ = diag(eigval_);
        Xform = eigvec(:,1:N);
        eigval_ = eigval_(1:N);
        eigval(idx,:) = eigval_';
        % 
        % keyboard
        % Check reconstruction error
        % channel_data_pca = Xform'*channel_data_zero_mean';
        % channel_data_rec = Xform*channel_data_pca;
        % err = sum((channel_data_rec-channel_data_zero_mean').^2,"all")        

        xforms(idx,:,:) = Xform;

    end
end
