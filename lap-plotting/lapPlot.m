%% reset system, initialize track points, see file for layout
clear;
clc;
trackData;
nPoints = max(size(Length));
x0=0;x1=0;y0=0;y1=0;angle=0;

%% loop through each section of lap
for n=1:1:nPoints
   L = Length(n);
   R = Radius(n);
   DIR = Type(n);
   %% turn the list of sections into fixed points for the start/end of each segment, 
   %% arcs will be plotted between each consecutive point later

   switch DIR
       %% depending on left/right turn, calculate new heading angle from previous heading angle
       case "Straight"
           %disp('str')
            x1 = L*cosd(angle) + x0;
            y1 = L*sind(angle) + y0;
            headingAng = angle;
            turnType = 0;

       case "Left"
            %disp('l')
            %% secant is direct path from start to end of segment
            secant = 2*R*sin(L/(2*R));

            %% heading angle is calculated as the tangent of the arc at the end of each segment
            headingAng = L*180/(pi*R)+angle;

            %% turning angle is calculated as the angle between the "entrance angle" and 
            %% the secant
            turnAng = L*90/(pi*R)+angle;

            %% turn type tells us whether it is CW or CCW, used in plotArc function
            turnType = L/R;
            angle = headingAng;

            %% coordinates for the endpoint of each segement
            x1 = secant*cosd(turnAng) + x0;
            y1 = secant*sind(turnAng) + y0;

       case "Right"
           %disp('r')
            secant = 2*R*sin(L/(2*R));
            headingAng = -L*180/(pi*R)+angle;
            turnAng = -L*90/(pi*R)+angle;
            turnType = L/R;
            angle = headingAng;
            x1 = secant*cosd(turnAng) + x0;
            y1 = secant*sind(turnAng) + y0;
            
       otherwise
           D = 0;
   end 
%    disp("turnType")
%    disp(turnType);
   %% create list of all points and segment data
   heading(n)=headingAng;
   turn(n)=turnType;
   ang(n)=angle; 
   x(n)=x1;y(n)=y1;
   x0 = x1;
   y0 = y1;
end

%% bookeeping for array indexing, nothing important
x = [0, x];
y = [0, y];

 for i=1:1:nPoints
     %% plot arcs in between consecutive points using the plotArc subfunction
     a = [x(i),y(i)];
      b = [x(i+1), y(i+1)];
      dir = Type(i);
      r = Radius(i);
      ang = turn(i);
      plotArc(a,b,r,dir,ang);
  end
  hold off

%% plot the track as line segments between consecutive points 
 figure;
 plot(x,y);
 hold on
 title('Track Points');
 axis equal;
 hold off