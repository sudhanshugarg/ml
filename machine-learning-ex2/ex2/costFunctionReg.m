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

f = size(X,2);
m = size(X,1);
for i=1:m
  if (y(i) == 1)
    J = J + log(sigmoid(X(i,:) * theta));
  else
    J = J + log(1 - sigmoid(X(i,:) * theta));
  endif
end

J = -1 * J/m;
J = J + lambda * sum(theta(2:end,:) .^ 2)/(2*m);

predicted = sigmoid(X * theta) .- y;
for i=2:f
  grad(i) = 1/m * sum(predicted .* X(:,i)) + lambda/m * theta(i);
end
grad(1) = 1/m * sum(predicted); %same as -> grad(1) = 1/m * sum(predicted .* X(:,1));



% =============================================================

end
