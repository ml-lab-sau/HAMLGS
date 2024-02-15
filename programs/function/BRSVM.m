
function [Pre_Labels, time] = BRSVM(train_data,train_target,test_data,test_target)
%BR The Binary Relevance method for MLC

% Training and testing by the libsvm
[num_class,num_test] = size(test_target);
[indx, v] = find(train_target(1,:)==0);
train_data(indx,:) = [];
train_target(:,indx) = [];
Pre_Labels = zeros(num_class,num_test);    
%Outputs = zeros(num_class,num_test);  
time=0;
for i = 1 : num_class
    train_label=train_target(i,:)';
    train_label(train_label~=1)=-1;
    test_label=test_target(i,:)';
    test_label(test_label~=1)=-1;
    
    % SVM 
    % C=0.1 default setting
    C = 0.1; ker = 'linear'; rr=1; kerpara=0.3;
    [w,b,~,time1] = svmclassifier(train_data,train_label,ker,C,rr,kerpara); 
    time = time + time1;
    z = test_data*w + b;
    Pre_Labels(i,:) = sign(z)';
    
%    % addpath('E:\rsch\MLML_Sylvester_CODE_AND_DEMO--BYWU_ICPR2015\functions\vlfeat-0.9.21\toolbox\vl_setup')
%     [w, b] = vl_svmtrain(train_data',train_label,0.1) ;
%     [~,~,~, scores] = vl_svmtrain(test_data', test_label, 0, 'model', w, 'bias', b, 'solver', 'none') ;
% Pre_Labels(i,:) = sign(scores)';%sign(predicted_label)';
% 
% %     [~, alpha, b] = svc(train_data,train_label,'rbf',0.1); 
% %     [Pre_Labels(i,:),Zstar(i,:)] = svcoutput(train_data,train_label,test_data,'rbf',alpha,b,0);
%     
% %    method.base.param.svmparam = '-s 2 -B 1 -q';%'-c 4 -t 0 -e 0.1 -m 800 -v 5';
% %     [model,method,time]=linear_svm_train(train_data,train_label,method);
% %     [conf,method,time]=linear_svm_test(train_data,train_label,test_data,model,method);
% %     Pre_Labels(i,:) = conf;
%     
%     %W(:,i) = w;
%     %B(i,:) = b;
end
Pre_Labels = Pre_Labels';
end

