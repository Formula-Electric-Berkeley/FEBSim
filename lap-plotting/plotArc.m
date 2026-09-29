function [output] = plotArc(p1,p2,radius,type, angle)
   A = transpose(p1); % Starting point
   B = transpose(p2); % Ending Point
   d = norm(B-A);
   R = radius; % Radius
   Ang = angle;
   switch R
       case 0
           plot([p1(1),p2(1)],[p1(2), p2(2)],'b-');
           hold on
       otherwise
           if type == "Right"
               C = real((B+A)/2-sqrt(R^2-d^2/4)/d*[0,-1;1,0]*(B-A)); % Center of circle
%                disp(C);
           else
               C = real((B+A)/2+sqrt(R^2-d^2/4)/d*[0,-1;1,0]*(B-A));
%                disp(C);
           end

           a = atan2(A(2)-C(2),A(1)-C(1));
           b = atan2(B(2)-C(2),B(1)-C(1));
           switch type
               case "Left"
                   b = mod(b-a,2*pi)+a; % for left turn
               case "Right"
                   b = -mod(a-b,2*pi)+a; % for right turn
               otherwise
                   disp("error")
           end
           t = linspace(a,b,1000);
           x = C(1)+R*cos(t);
           y = C(2)+R*sin(t);
           plot(x,y,'b-',C(1),C(2),'w*')
           title("FEB 2012 Nebraska Endurance Track")
           axis equal
           hold on
   end
end