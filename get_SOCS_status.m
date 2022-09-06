function [] = get_SOCS_status()

src_cse = read_PISM2SOCS_input('SOCS_SRC_CSE');
src_pth = realpath(read_PISM2SOCS_input('SOCS_SRC_OUT_PTH'));
s2p_var = read_PISM2SOCS_input('S2P_VAR_SRC');

mod_prm= data_case_V04(src_pth,'time',s2p_var,src_cse);
if isempty(mod_prm)
    
    Nyr=0;
    disp('0|0|0|0')
    
else
    % choose last time step
    
    [~,Ti]=calendar_index(mod_prm.src_tme,mod_prm.src_dpyr,mod_prm.src_ref_dte);

    if ceil(Ti.y(end))-Ti.y(end) < 0.05
        Nyr = double(ceil(Ti.y(end)))-1;
        % -1 because the accomplished year is one less than the calendar
        % shows i.e. time step Jan 2013 indicates year 2012 is accomplished
    else
        Nyr = double(floor(Ti.y(end)))-1;
    end
    
    % which output is closest to the actual year end of Nyr
    mod_tme = mod_prm.src_tme+datenum(mod_prm.src_ref_dte);
    trg_tme = datenum([Nyr+1 1 0]);
    [~,i]=min(abs(mod_tme-trg_tme));
    
    if ceil(Ti.y(1))-Ti.y(1) < 0.05
        yr0 = double(ceil(Ti.y(1)))-1;
    else
        yr0 = double(floor(Ti.y(1)))-1;
    end
    disp([ num2str(yr0)  '|' num2str(Nyr) '|' src_pth '/' char(mod_prm.fle_nmes(mod_prm.fle_ind(i))) '|' num2str(mod_prm.nct_ind(i))])
    
    
end

end

