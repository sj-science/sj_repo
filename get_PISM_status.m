function [] = get_PISM_status()

src_fnm = read_PISM2SOCS_input('PISM_SRC_CSE');
src_pth = realpath(read_PISM2SOCS_input('PISM_SRC_PTH'));
p2s_var = read_PISM2SOCS_input('P2S_VAR_SRC');

mod_prm= data_case_V04(src_pth,'time',p2s_var,src_fnm);
if isempty(mod_prm)
    
    Nyr=0;
    disp('0|0|0|0')
else
    % choose last time step
    
    [~,Ti]=calendar_index(mod_prm.src_tme,mod_prm.src_dpyr,mod_prm.src_ref_dte);
   
        Nyr = double(ceil(Ti.y(end)))-1;
        % -1 because the accomplished year is one less than the calendar
        % shows i.e. time step Jan 2013 indicates year 2012 is accomplished
    
    % which output is closest to the actual year end of Nyr
    mod_tme = mod_prm.src_tme+datenum(mod_prm.src_ref_dte);
    trg_tme = datenum([Nyr+1 1 0]);
    [~,i]=min(abs(mod_tme-trg_tme));
    
    %disp([num2str(ceil(Ti.y(1)))  '|' num2str(Nyr) '|' src_pth '/' char(mod_prm.fle_nmes(mod_prm.fle_ind(i))) '|' num2str(mod_prm.nct_ind(i))])
    disp([num2str(ceil(Ti.y(1)))  '|' num2str(Nyr) '|' char(mod_prm.fle_nmes(mod_prm.fle_ind(i))) '|' num2str(mod_prm.nct_ind(i))])
    
end

end

