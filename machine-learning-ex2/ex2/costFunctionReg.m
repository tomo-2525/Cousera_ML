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


%目的関数を求める
for i = 1:m
    z = X(i,:) * theta;
    h = sigmoid(z);
    J += -y(i) *  log(h) - (1 - y(i))*log(1-h);
end
J = J/m;
%上のコードは、コストファンクションからコピペしたものです


%追加
for j = 2:size(theta) %θ0には、ペナルティを与えない
    J +=  theta(j) * theta(j) * ( lambda / (2*m)  ) ;
end 



%ロジスティック回帰の最急降下法に使う偏微分を求める

for j = 1 : size(theta)
     for i = 1 : m
    z = X(i,:) * theta;
    h = sigmoid(z);

    grad(j) +=  (h - y(i) ) * X(i,j);

     end 
     if j > 1  %θ０には、ペナルティを与えない？
         grad(j) += lambda*theta(j);
     end 
end
     
grad = grad/m;



% =============================================================

end
