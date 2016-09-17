function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%Cost Function non regularized
h = X*theta;
error = y - h;
error_sqr = (y - h).^2;
J = sum(error_sqr)/(2*m);

%Gradient non regularized

grad = -1*(X'*error)/m;

theta(1) = 0;

%Cost regularized component
J_reg = (sum(theta.^2))*lambda/(2*m);

%Gradient regularized component
grad_reg = theta*(lambda/m);

%Cost Function 
J = J + J_reg;

%Grad Function
grad = grad + grad_reg;

% =========================================================================

grad = grad(:);

end
