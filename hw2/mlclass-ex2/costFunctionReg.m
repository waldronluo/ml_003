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

% Hyprophsis Function
function h = H (X, theta)
    h = sigmoid(X * theta);
end
% End of Hyprophsis Function

for j = 1:size(theta) 
    Sum = 0;
    for i = 1:m
        Sum += (H(X(i,:), theta) - y(i)) * X(i,j);
    endfor
    grad(j) = Sum / m + (j != 1) * lambda * theta(j) / m;
endfor


Sum = 0;
for i = 1:m 
    Sum += ( -y(i)*log(H( X(i,:), theta)) - (1 - y(i))*log(1 - H(X(i,:), theta)));    
endfor
J = Sum / m + lambda * (theta' * theta) / (2 * m);
J
% =============================================================

end
