function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h=zeros(m,1);
h=sigmoid(X*theta);

cost_sum= (1/m)*((-1*y'*log(h))-((1-y')*log(1-h)));
cost_reg= lambda*(1/(2*m))*(sum(theta.^2)-(theta(1,1)^2));
# need to fix cost_reg to not regularize for theta(0)
J=cost_sum+cost_reg;

for i=1:length(grad)
  err=(h-y).*X(:,i);
  if i>1 
    grad(i)=(1/m)*sum(err) + (lambda/m)*theta(i);
   else
    grad(i)=(1/m)*sum(err);
end




% =============================================================

end
