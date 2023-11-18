function [result]=AMSLOD(X,lambda1,lambda2,omega,gt)
fprintf("AMSLOD is running:\n")
lambda3 = 0.1;
epson = 1e-7;
C = size(unique(gt),1); % number of clusters
V = size(X,2); % number of views
N = size(X{1},2);% number of data points
%normalized X
for i=1:V
    X{i} = X{i}./repmat(sqrt(sum(X{i}.^2,1)),size(X{i},1),1);
end
%Initilize Z,E,tensor G,multiplier Y,W
K =80;
for i = 1:V
    Z{i} = constructW_PKN(X{i},10);
    P{i} = rand(size(X{i},1),K);
    H{i} = zeros(K,N);
    Y{i} = zeros(size(X{i},1),N);
    G{i} = zeros(N,N);
    W{i} = zeros(N,N);  %multiplier
end

weight = 1/V*ones(1,V);  %  adaptive weight
sX = [N, N, V];
pho = 0.1;
pho_max = 1e10;
eta = 2;
converge_Z_G=[];

Isconverg = 0;
iter = 1;
%%  iteration
while(Isconverg == 0)
    % == update F,A ==
    if iter ==1
        [Y1, A] = CLR(weight,Z,C,lambda3);
    else
        [Y1, A] = CLR(weight,Z,C,lambda3,A0);
    end
    A0 = A;  %after iter=1,we have S0


    % == update Z{i} ==
    for i = 1:V
        B1 = G{i}-W{i}./pho;
        Z{i} = inv(2*lambda2*H{i}'*H{i}+2*weight(1,i)*eye(N,N)+pho*eye(N,N))*(2*lambda2*H{i}'*H{i}+2*weight(1,i)*A+pho*B1);
    end

    % == update H{i} ==
    for i = 1:V
        A1 = lambda1*P{i}'*P{i};
        A2 = lambda2*eye(N,N)+lambda2*Z{i}*Z{i}'-lambda2*(Z{i}+Z{i}');
        A3 = -1*lambda1*P{i}'*X{i};
        H{i} = lyap(A1,A2,A3);
    end

    % == update G{i} ==
    Z_tensor = cat(3, Z{:,:});
    W_tensor = cat(3, W{:,:});
    z = Z_tensor(:);
    w = W_tensor(:);
    [g, ~] = wshrinkObj(z+1/pho*w,1/pho,sX,0,3,omega);
    G_tensor = reshape(g, sX);
    for i=1:V
        G{i} = G_tensor(:,:,i);
    end
    % == update P{i} ==
    for i=1:V
        G1 = H{i}';
        Q1 = X{i}';
        W1 = G1'*Q1;
        [U1,~,V1] = svd (W1,'econ');
        PT = U1*V1';
        P{i} = PT';
    end

    % update weight
    for i = 1:V
        weight(1,i) = 0.5/norm(A-Z{i},'fro');
        %weight(1,i) = 1/V;
    end

    % == update W{i} ==
    for i=1:V
        W{i} = W{i}+pho*(Z{i}-G{i});
    end

    max_Z_G=0;
    Isconverg = 1;
    for k = 1:V

        if (norm(Z{k} - G{k}, inf) > epson)
            history.norm_Z_G = norm(Z{k} - G{k}, inf);
            Isconverg = 0;
            max_Z_G = max(max_Z_G, history.norm_Z_G);
        end

    end
    converge_Z_G=[converge_Z_G max_Z_G];


    % == update pho  mu ==
    pho = min(pho_max,pho*eta);
    iter = iter + 1;
    if (iter==50)
        Isconverg = 1;
    end

end

[ACC,NMI,PUR] = ClusteringMeasure(gt,Y1); %ACC NMI Purity
[Fscore,Precision,R] = compute_f(gt,Y1);
[AR,~,~,~]=RandIndex(gt,Y1);
result = [ACC NMI PUR Fscore Precision R AR];
