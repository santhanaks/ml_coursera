function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


# set y = binary values
y=eye(num_labels)(y,:);

X=[ones(m,1) X]; #add ones to X

z2=zeros(m,size(Theta1,1));
z2=X*Theta1';
a2=sigmoid(z2); # compute activations for hidden layer

a2=[ones(m,1) a2];
h=zeros(m,size(Theta2,1)); # initialize output layer
h=sigmoid(a2*Theta2'); # compute output layer

cost=0;

for i=1:m
   y_i=zeros(num_labels,1);
   h_i=zeros(1,num_labels);
   y_i=y(i,:);
   h_i=h(i,:);
   sumOverK=sum((-1*y_i.*log(h_i))-((1-y_i).*log(1-h_i)));
   cost=cost+sumOverK;
end
J=(1/m)*cost;

% Regularization for cost function start
  Theta1FirstCol=sum(Theta1(:,1).^2);
  Theta2FirstCol=sum(Theta2(:,1).^2);
  Theta1Reg=sum(sum(Theta1.^2))-Theta1FirstCol;
  Theta2Reg=sum(sum(Theta2.^2))-Theta2FirstCol;
  J=J+((lambda/(2*m))*(Theta1Reg+Theta2Reg));
% -------------------------------------------------------------
% Back Propagation Start


a1=X;
Delta2=zeros(num_labels,size(Theta2,2));
#printf("size of a1 = %ix%i\n",size(a1,1),size(a1,2));
#printf("size of h = %ix%i\n",size(h,1),size(h,2));
#printf("size of y = %ix%i\n",size(y,1),size(y,2));
delta3=h-y;
#printf("size of delta3 = %ix%i\n",size(delta3,1),size(delta3,2));
#printf("size of a2 = %ix%i\n",size(a2,1),size(a2,2));
Delta2=Delta2+delta3'*a2;
#printf("size of z2 = %ix%i\n",size(z2,1),size(z2,2));
#printf("size of Delta2 = %ix%i\n",size(Delta2,1),size(Delta2,2));

Delta1=zeros(size(Theta1,1),size(X,2));
delta2=(delta3*Theta2(:,2:end)).*sigmoidGradient(z2);
#printf("size of delta2 = %ix%i\n",size(delta2,1),size(delta2,2));
Delta1=Delta1+delta2'*X;
#printf("size of Delta1 = %ix%i\n",size(Delta1,1),size(Delta1,2));

Theta1_grad=(1/m)*Delta1;
Theta2_grad=(1/m)*Delta2;


Theta1(:,1)=0;
Theta2(:,1)=0;
Theta1=(lambda/m)*Theta1;
Theta2=(lambda/m)*Theta2;
Theta1_grad=Theta1_grad+Theta1;
Theta2_grad=Theta2_grad+Theta2;
% =========================================================================
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
