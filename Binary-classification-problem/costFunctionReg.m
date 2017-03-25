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
h=X*theta;
g = zeros(size(h));
for i = 1:rows(h),
  for j = 1:columns(h),
g(i,j)=1/(1+(e^(-1*h(i,j))));
end;
end;
h=g;
sum=0;
for i=2:size(theta),
sum= sum + theta(i, 1)^2;
end;
J=(-1/m)*(y'*log(h)+(1-y)'*log(1-h)); 
J= J + (lambda/(2*m))*sum;


grad=(1/m)*(X'*(h-y)) + (lambda/m)*theta;
grad(1)=grad(1)- (lambda/m)*theta(1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
