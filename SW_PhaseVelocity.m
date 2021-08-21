% Load shear wave traces from verasonics/siemens imaging system
%load('SWS Data');
Fs = FPS;            % Sampling frequency                    
T = 1/Fs;             % Sampling period       
n=1024;
%wave=wave{1};
L = size(wave,3);             % Length of signal
t = (0:L-1)*T; 
x1=x;z1=z;
%1D fourier transform V(z,x,t)----V~(z,x,f)
for l1=1:size(wave,1)
    for l2=1:size(wave,2)
        push1=permute(wave(l1,l2,:),[3 2 1]);
        push1_f=(fft(push1,n))/L;
        push1_fabs = (push1_f);
        %push1_fabs = abs(push1_f/n);
        P1 = push1_fabs(1:((n/2+1)));
        %P1 = push1_fabs(1:L.2+1);
       % P1(2:end-1) = 2*P1(2:end-1);
        push1_fourier_int(l1,l2,:)=P1;
        %push1_fourier(l1,l2,:)=push1;
        warning('off','all')
    end
end
% find corresponding frequency components
%f = Fs*(0:(L/2))/L;
f = Fs*(0:(n/2))/n;
%f = Fs*(0:(n-1))/n;
%rearrange lateral pixel positions so that origin is at left corner
%x1=x1-x1(1,1); %uncomment for experimental data

%local window
winlength1=15;%21
winlength2=15;%15

x11=0:x1(1,end)/(size(push1_fourier_int,2)-1):x1(1,end);
z11=0:z1(1,end)/(size(push1_fourier_int,1)-1):z1(1,end);

Fs_x = 1000*size(x11,2)/(x11(1,end));       % pixels per milimeter
        Fs_z = 1000*size(z11,2)/(z11(1,end)-z11(1,1)); 
        dx1 = 1/Fs_x;     % meters per pixel
        dz1 = 1/Fs_z;
        count=0;
for freq=1:3:100%1:4:size(f,2)%1:3:260%1:2:70 for comsol % 1:3:260 for liver
    count=count+1;
for wr=1:size(push1_fourier_int,1)
    for wc=1:size(push1_fourier_int,2)
        wx_pos=wr-(winlength1-1)/2:wr+(winlength1-1)/2;
        wy_pos=wc-(winlength2-1)/2:wc+(winlength2-1)/2;
        if (wr-(winlength1-1)/2)<=0
            wx_pos=1:wr+(winlength1-1)/2;
        end
        if (wc-(winlength2-1)/2)<=0
            wy_pos=1:wc+(winlength2-1)/2;
        end
        if (wr+(winlength1-1)/2)>size(push1_fourier_int,1)
            wx_pos=wr-(winlength1-1)/2:size(push1_fourier_int,1);
        end
        if (wc+(winlength2-1)/2)>size(push1_fourier_int,2)
            wy_pos=wc-(winlength2-1)/2:size(push1_fourier_int,2);
        end
        
        %determining kernel positions in mm
        x_x1=x11(1,wy_pos(1,1):wy_pos(1,end));
        z_z1=z11(1,wx_pos(1,1):wx_pos(1,end));
        
        weight_x=gpuArray(zeros(1,size(x_x1,2)));
        weight_z=gpuArray(zeros(1,size(z_z1,2)));
        
        %tapering coefficient
        a=0.25;
        %x axis tuckey window
        if abs(x_x1(1,end)-x_x1(1,1))~=2*abs(x_x1(1,1)-x11(1,wc))
            mag=2*max(abs(x_x1(1,1)-x11(1,wc)),abs(x_x1(1,end)-x11(1,wc)));
        else
            mag=abs(x_x1(1,end)-x_x1(1,1));
        end
      
        for dl=1:size(x_x1,2)
            req=abs(x_x1(1,dl)-x11(1,wc));
            if req>(mag)
                weight_x(1,dl)=0;
            elseif req>=0 && req<(a.*mag)
                weight_x(1,dl)=1;
            elseif req>=(a.*mag) && req<=(mag)
                arg=pi.*(x_x1(1,dl)-x11(1,wc)-(a.*mag))./(2.*(1-a).*mag);
                weight_x(1,dl)=0.5.*(1+cos(arg));
            end
        end
        
        %z-axis tuckey window
        if abs(z_z1(1,end)-z_z1(1,1))~=2*abs(z_z1(1,1)-z11(1,wr))
            mag1=2*max(abs(z_z1(1,1)-z11(1,wr)),abs(z_z1(1,end)-z11(1,wr)));
        else
            mag1=abs(z_z1(1,end)-z_z1(1,1));
        end
        
        for dl=1:size(z_z1,2)
            req1=abs(z_z1(1,dl)-z11(1,wr));
            if req1>(mag1)
                weight_z(1,dl)=0;
            elseif (req1>=0) && (req1<(a.*mag1))
                weight_z(1,dl)=1;
            elseif req1>=(a.*mag1) && req1<=(mag1)
                arg=pi.*(z_z1(1,dl)-z11(1,wr)-(a.*mag1))./(2.*(1-a).*mag1);
                weight_z(1,dl)=0.5.*(1+cos(arg));
            end
        end
        
        Fx= (-Fs_x/2:(Fs_x)/(n-1):Fs_x/2)';
        Fz = (-Fs_z/2:(Fs_z)/(n-1):Fs_z/2)';
        %common tuckey window multiplied by V(z,x,f) to get V*(z,x,f)
       weight=zeros(size(weight_z,2),size(weight_x,2));
   
       weight=weight_z'.*weight_x;
       
       %V_star1(1:size(weight,1),1:size(weight,2),freq)=zeros(size(weight,1),size(weight,2));
        %k_abs=zeros(1,size(push1_fourier_int,3));
       %for freq=1:size(push1_fourier_int,3)                 
          % compute for all frequencies of shear wave data               
        temp=gpuArray(weight(:,:)).*gpuArray(push1_fourier_int(wx_pos(1,1)+1-1:wx_pos(1,1)+size(weight,1)-1,wy_pos(1,1)+1-1:wy_pos(1,1)+size(weight,2)-1,freq));
        V_star1=gpuArray(temp);
          %Fourier of short space
         spectrum=(fft2(V_star1,n,n));  spectrum(1,1)=0;
         
         im_ac=ifft2(spectrum);
        V_spec1=(fftshift(fft2(im_ac,n,n))); 
        
        [sd sd2]=find(V_spec1==max(max((V_spec1))));
        Fx_m=Fx(sd2,1);Fz_m=Fz(sd,1);
        kx_m=2*pi*Fx_m;
        kz_m=2*pi*Fz_m;       
        k_abs=abs(sqrt((kx_m.^2+kz_m.^2)));
        
        phase_vel=2*pi*f(1,freq)/k_abs;
        if isempty(phase_vel)==1
            phase_vel=0;k_abs=0;
        end
        p_v(wr,wc,count)=phase_vel;   
        k_mag(wr,wc,count)=k_abs;
    end

end
count
seq_f(1,count)=freq;
save('phase_velocity.mat','p_v','k_mag','f','seq_f');

end



