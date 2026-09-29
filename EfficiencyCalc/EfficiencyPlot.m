function Plot = EfficiencyPlot(OLResults, EffImg)    
    %plots a OptimumLap run over the Emrax 228 Efficiency graph to better
    %understand motor efficiency visually.

    y_max = 1417;
    y_min = 477;
    x_max = 1582;
    x_min = 157;
    TorqueRange = 250;
    RPMRange = 5000;
    
    
    
    img = imread(EffImg);
    imshow(img);
    %axis = [0,0,1000,2500];
    hold on;
    
    
    RPM = OLResults(:,2);
    Torque = OLResults(:,3);
    x = y_max - (Torque/TorqueRange)*(y_max-y_min);
    y = x_min + (RPM*60/2/pi/RPMRange)*(x_max-x_min);
    plot(y,x,'b*');
    
    %{
    for n = 1:8
        RPM = OLResults(:,4*n-2);
        Torque = OLResults(:,4*n-1);
        x = y_max - (Torque/TorqueRange)*(y_max-y_min);
        y = x_min + (RPM*60/2/pi/RPMRange)*(x_max-x_min);
        switch n
            case 1
                plot(y,x,'r*');
            case 2
                plot(y,x,'g*');
            case 3
                plot(y,x,'b*');
            case 4
                plot(y,x,'y*');
            case 5
                plot(y,x,'c*');
            case 6
                plot(y,x,'m*');
            case 7
                plot(y,x,'w*');
            case 8
                plot(y,x,'k*');
        end
    end
    %}
    
end