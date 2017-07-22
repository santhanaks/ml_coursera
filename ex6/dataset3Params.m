function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
PossibleDataSet=[0.01 0.03 0.1 0.3 1 3 10 30];
num_elements=numel(PossibleDataSet);
errors=zeros(1,num_elements^2);

for C_select=1:num_elements
  C_input=PossibleDataSet(1,C_select);
  for sigma_select=1:num_elements
    sigma_input=PossibleDataSet(1,sigma_select);
    model= svmTrain(X, y, C_input, @(x1, x2) gaussianKernel(x1, x2, sigma_input));
    predictions=svmPredict(model,Xval);
    errors(1,num_elements*(C_select-1)+sigma_select)=mean(double(predictions~=yval));
  end
end

min_location=find(errors==min(errors));
#printf("min_location = %i\n",min_location);
C=PossibleDataSet(fix(min_location/num_elements));
#printf("C = %d\n",C);
sigma=PossibleDataSet(rem(min_location,num_elements));
#printf("sigma = %d\n",sigma);
% =========================================================================

end
