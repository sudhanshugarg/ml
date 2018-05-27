function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%


m = size(X,1);% number of training examples
f = size(X,2);% number of features, same as size(theta, 1);
%cost function = 1/m summation i=1tom [-yi * log(ht(xi)) - (1-yi) * log(1-ht(xi))]
%gradient => thetaJ = thetaJ - alpha/m * summation i=1tom (ht(xi) - yi) * xij
%J is a scalar, grad is a fx1 vector, capiche?

for i=1:m
  if (y(i) == 1)
    J = J - log(sigmoid(X(i,:) * theta));
  else
    J = J - log(1 - sigmoid(X(i,:) * theta));
  endif
end
J = J/m;

for i=1:f
  grad(i) = 1/m * sum((sigmoid(X*theta) .- y) .* X(:,i));
end

% =============================================================

end
