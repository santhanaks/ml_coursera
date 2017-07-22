function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. This should work regardless if z is a matrix or a
%   vector. In particular, if z is a vector or matrix, you should return
%   the gradient for each element.

g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the gradient of the sigmoid function evaluated at
%               each value of z (z can be a matrix, vector or scalar).
zlength=size(z,1);
zwidth=size(z,2);
unRollz=[z(:)];
g=sigmoid(unRollz).*(1-sigmoid(unRollz));
g=reshape(g(1:zlength*zwidth),zlength,zwidth);
% =============================================================
end
