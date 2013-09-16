function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               Y ou should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% Internal Function
function h = H (X, theta)
    h = sigmoid(X * theta);
end
% End of Internal Function

alpha = 0.1;

for j = 1:size(theta)
    Sum = 0;
    for i = 1:m
        Sum += (H(X(i,:), theta) - y(i)) * X(i,j);
    endfor
    grad(j) = Sum / m;
endfor

Sum = 0;
for i = 1:m 
    Sum +=((-y(i) * log(H(X(i,:), theta))) - (1 - y(i)) * log(1-H(X(i,:), theta)));
endfor

J = 1 / m * Sum;

J
grad
% =============================================================

end


