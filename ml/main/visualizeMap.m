function visualizeMap(dataArray,userYlabel,xDim,step,rotate,varargin)

    if nargin == 5
        dataMat = rot90(reshape(dataArray,[],xDim));
    else
        dataMat = reshape(dataArray,[],xDim);
    end
    
    [n,m] = size(dataMat);
    [XX,YY] = meshgrid((0:m-1)*step, (0:n-1)*step);
    
    figure();
    pcolor(XX,YY,dataMat);
    
    hc = colorbar;
    hc.Label.String=userYlabel;
    
    colormap('turbo')
    %colormap(flipud(turbo))
    
    shading interp
    axis equal
    axis tight
    xlabel('Position (mm)')
    ylabel('Position (mm)')
    set(gca,'FontSize',16)
    
    %xtickMark=linspace(0,2,7);
    %tickMark=0:0.4:1.6;
    %xticks(tickMark);
    %yticks(tickMark);
    
    %exportgraphics(gca,'cap3.png')
