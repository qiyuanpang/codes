if(1)
    K=9; %number of waves
    NPW=9; % numer of grids per wave
    NPML=8; % number of PMLs at the boundary of the domain
    CPML=10; % PML parameter for boundary PMLs
    omega = 2*pi*K;
    
    %num of grid points
    NC = K*NPW; %number of cells
    h = 1/(NC+1);
    N=NC; N1=N;N2=N;
    Nsample = 2048;%25000
    
    %grid and window
    [x1,x2]=ndgrid(h*(1:N));    x1 = x1-1/2;    x2 = x2-1/2;
    %dis = min(1/2-abs(xx),1/2-abs(yy));    win = 1-exp(-(dis)^2);
    
    %PML
    gs=(1/NPML)*(0.5:0.5:NPML-0.5); eta=NPML*h;
    sigR=CPML/eta*gs.^2;
    sR=1./(1+1i*sigR/K);sL=sR(end:-1:1);
    s1=[sL,ones(1,2*(N1-2*(NPML-1))-1),sR];
    s2=[sL,ones(1,2*(N2-2*(NPML-1))-1),sR];    %nb = 2*NPML;
    
    %num directions
    Ndir = K*NPW;
    thetas = [0:Ndir-1]/Ndir * 2*pi;
    UI = zeros(N,N,Ndir);
    for b=1:Ndir
        tmp = thetas(b);
        t1 = cos(tmp);        t2 = sin(tmp);
        uinc = exp(i*omega*(x1*t1 + x2*t2));
        UI(:,:,b) = uinc;
    end
    UI = reshape(UI, N^2, Ndir);
    
    coes = zeros(N,N,Nsample);
    sols = zeros(Ndir,Ndir,Nsample);
    adjs = zeros(N,N,Nsample);
    
    gaussian = @(u1,u2,T,x1,x2) exp(-((x1-u1).^2+(x2-u2).^2) / (2*T));
    mag = 0.2;
    for idx=1:Nsample
        idx
        %get media
        ng = 4;
        aux = zeros(size(x1));
        for gi=1:ng
            a=-0.3;            b=0.3;            u1 = a + rand(1)*(b-a);
            a=-0.3;            b=0.3;            u2 = a + rand(1)*(b-a);
            aux = aux + mag*gaussian(u1,u2,0.015^2,x1,x2);
        end
        m = aux;
        eta = -omega^2*m; %KEY: eta is the coefficient
        
        %get matrix
        c = sqrt(1./(1-m));
        ksq=(omega./c).^2;
        A=setupL2d(h,ksq,s1,s2);
        
        %solve
        DS = spdiags(eta(:), 0, N^2,N^2);
        U = A\(DS*UI);
        UT = U + UI; %total wave
        R = 1/(2*pi)*UI'*(DS*UT) * h^2;
        
        %store data
        coes(:,:,idx) = eta;
        sols(:,:,idx) = R;
    end
    
    [pa,qa] = ndgrid(thetas);
    p1 = cos(pa);    p2 = sin(pa);
    q1 = cos(qa);    q2 = sin(qa);
    p1 = p1(:);    p2 = p2(:);
    q1 = q1(:);    q2 = q2(:);
    M = 1/(2*pi) * (exp(-i*omega*((p1-q1).*x1(:)'+(p2-q2)*x2(:)') ));
    
    adjs = (M'*(2*pi/Ndir)^2) * reshape(sols,[Ndir^2,Nsample]);
    adjs = reshape(adjs, [N,N,Nsample]);
    if 1
    for idx=1:Nsample
        subplot(1,3,1); imagesc(coes(:,:,idx)); colorbar;
        subplot(1,3,2); imagesc(real(sols(:,:,idx))); colorbar;
        subplot(1,3,3); imagesc(real(adjs(:,:,idx))); colorbar;
        pause(0.5);
    end
    end
end


if 1

coes = coes - mean(coes(:));
sols = sols - mean(sols(:));
adjs = adjs - mean(adjs(:));
 

coes = coes/(max(abs(coes(:))));
sols = sols/(max(abs(sols(:))));
adjs = adjs/(max(abs(adjs(:))));

end

%fwd data
if(1)
    suffix = 'data/scafull2';
    fileinput  = [suffix, '.h5'];
    if exist(fileinput, 'file') == 2
        delete(fileinput);
    end
    h5create(fileinput, '/Input', [N,N, Nsample]);
    h5write( fileinput, '/Input', real(coes)); 
    h5create(fileinput, '/Output', [Ndir,Ndir, Nsample]);
    h5write( fileinput, '/Output', real(sols));
    h5create(fileinput, '/Adjoint', [N,N, Nsample]);
    h5write( fileinput, '/Adjoint', real(adjs));
    h5create(fileinput, '/Input2', [N,N, Nsample]);
    h5write( fileinput, '/Input2', imag(coes)); 
    h5create(fileinput, '/Output2', [Ndir,Ndir, Nsample]);
    h5write( fileinput, '/Output2', imag(sols));
    h5create(fileinput, '/Adjoint2', [N,N, Nsample]);
    h5write( fileinput, '/Adjoint2', imag(adjs));
end
    
if(0)
    %c = ones(size(xx))*2;
    c=ones(size(xx))-0.4*exp(-1024*(xx.^2+yy.^2));
    %c=ones(size(xx))+0.4*exp(-1024*(xx.^2+yy.^2));
    %c = ones(size(xx));    gd = find(xx.^2/0.1^2+yy.^2/0.1.^2<1); c(gd) = 1.5;
    %c = ones(size(xx));    gd = find(xx.^2/0.3^2+yy.^2/0.3.^2<1); c(gd) = 1.5;
    c = ones(size(xx));    gd = find(xx.^2/0.3^2+yy.^2/0.1.^2<1); c(gd) = 1.1;
    %c = ones(size(xx));    gd = find(max(abs(xx),abs(yy))<0.3); c(gd) = 1.5;
    c = ones(size(xx));    gd = find(abs(xx-0.1)<0.3&abs(yy)<0.02); c(gd) = 0.5;
end

