%% This code is modeling the individual farmer maximize the profit subject to the pollutant cumulation by
% choosing the mulch decay rate. Given the optimal disposal strategy is to
% dispose of all the pre-tilled mulches



function Z1policy0P3
%% FORMULATION
 
format long

% Model parameters
a = [18000 1 0.1 0.005 5 0];
b = [0 0 1237.56 -0.5];
nu =[18621.39 10];
%ita 1 was 3.20, but we added 0.05 of residue which is A base level of residue from PE (or BDM for that matter) whether or not the mulch goes to a landfill

ita = [3.25 0.5 0.1 3.2 -0.5];
delta = 0.9;
w = 0.04;
p = 3;
z = 1;
             	
  
% Model structure
model.func = @func;                             % model functions
model.params = {a,b,nu,ita,z,w,p};              % function parameters
model.discount = delta;                         % discount factor
model.ds = 1;                                   % dimension of continuous state
model.dx = 1;                                   % dimension of continuous action
model.ni = 0;                                   % number of discrete states
model.nj = 0;                                   % number of discrete actions

% Approximation structure
n    = 100;                                     	% number of collocation nodes
smin = 0;                                       	% minimum state
smax = 50;                                          % maximum state
basis = fundefn('spli',n,smin,smax);        
%% SOLUTION
  
% Steady-state
format long
qstar =((nu(2)+w*(-b(2)+b(4)*z)+a(2)*p)*(1-delta*ita(3))*(1-ita(3))+(a(6)...
    *(1-delta*ita(3))+delta*a(5)*(-ita(2)-ita(5)*z))*p*(ita(1)-ita(4)*z)+...
    a(3)*delta*(-ita(2)-ita(5)*z)*p*(1-ita(3)))/...
    ((-a(4)*(1-ita(3)*delta)+a(6)*delta*(ita(2)+ita(5)*z))*p*(1-ita(3))+(-a(6)*(1-delta*ita(3))...
    +(a(5)*delta*(ita(2)+ita(5)*z))*p*(-ita(2)-ita(5)*z))); 	% choice variable
sstar = (ita(1)-ita(2)*qstar-ita(4)*z-ita(5)*z*qstar)/(1-ita(3));           % state variable
lstar = p*(-a(3)-a(5)*sstar-a(6)*qstar)/(-delta*ita(3)+1);                            % shadow price
pistar = (p*(a(1)-a(2)*qstar-a(3)*sstar-0.5*a(4)*qstar.^2-0.5*a(5)*sstar.^2-a(6)*qstar.*sstar)...
        -nu(1)-nu(2)*qstar-w*(b(1)-b(2)*qstar+b(3)*z+b(4)*z*qstar))/(1-delta); % The present value of life time profit givern the optimal choice
% check the condition that the optimal z=1, check value>0
check = -w*(b(3)+b(4)*qstar)+delta*(-ita(4)-ita(5)*qstar)*lstar;
fprintf('check the condition that the optimal z=1, check value>0 %5.2f\n' , check)
% Chekc the condition that the FOC=0
check1=-a(2)*p-a(4)*qstar*p-a(6)*sstar*p-nu(2)+w*b(2)-b(4)*z*w+delta*lstar*(-ita(2)-ita(5)*z);
fprintf('Chekc the condition that the FOC=0 %5.2f\n' , check1)

% Check model derivatives
dpcheck(model,sstar,qstar);
% Solve collocation equation
[c,s,v,q,resid] = dpsolve(model,basis);

max(v)



%% DPSOLVE FUNCTION FILE


function [out1,out2,out3] = func(flag,s,q,~,~,~,a,b,nu,ita,z,w,p)

switch flag
  case 'b'      % bounds
    out1 = zeros(size(s));
    out2 = zeros(size(s));
    out3 = [];
  case 'f'      % Profit
    out1 = p*(a(1)-a(2)*q-a(3)*s-0.5*a(4)*q.^2-0.5*a(5)*s.^2-a(6)*q.*s)...
        -nu(1)-nu(2)*q-w*(b(1)-b(2)*q+b(3)*z+b(4)*z*q);
    out2 = -a(2)*p-a(4)*q*p-a(6)*s*p-nu(2)+w*b(2)-b(4)*z*w;
    out3 =  -a(4)*p*ones(size(s));
  case 'g'      % Motion of state variable
    out1 = ita(1)-ita(2)*q+ita(3)*s-ita(4)*z-ita(5)*q*z;
    out2 = (-ita(2)-ita(5)*z)*ones(size(s));
    out3 = zeros(size(s));
end
