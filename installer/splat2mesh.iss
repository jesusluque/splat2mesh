; splat2mesh for Windows, everything local (NVIDIA GPU, 8 GB or more). Inno Setup 6.3 or later.
; Built by installer\build.ps1 from a staging folder (the repository's code, no outputs).
;
; Wizard: licence → folder → components → Idol's VAE → Iris-3B mode (from the detected GPU) → keys → install.
; The install step copies the code, then runs installer\bootstrap.ps1 (uv, Python 3.12, the package) and
; "python -m splat2mesh.winsetup install ..." for each chosen component, with a progress bar fed by their
; PROGRESS lines. Tokens go to the child processes through temporary files they delete at once.

#define AppName "splat2mesh"
#ifndef AppVersion
  #define AppVersion "0.4.0"
#endif
#ifndef Staging
  #define Staging "staging"
#endif

[Setup]
AppId={{5C3B7E52-8E1D-4C1B-9B7A-2F5E8D3A51C0}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=Lucab
AppPublisherURL=https://splat2mesh.lucab.co.uk
AppSupportURL=https://github.com/jesusluque/splat2mesh
DefaultDirName={localappdata}\splat2mesh
DefaultGroupName=splat2mesh
PrivilegesRequired=lowest
DisableProgramGroupPage=yes
LicenseFile={#Staging}\LICENSE
OutputDir=output
OutputBaseFilename=splat2mesh-{#AppVersion}-windows-setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.19041
UninstallDisplayName=splat2mesh

[Types]
Name: "full"; Description: "Everything (sentence → picture → splat → materials → USD → renders)"
Name: "gpu8"; Description: "Everything except local Iris-3B (8 GB GPUs: sentences through the online Space)"
Name: "render"; Description: "Renderer only (existing splats → USD, turntables)"
Name: "custom"; Description: "Custom"; Flags: iscustom

[Components]
Name: "core"; Description: "splat2mesh (Python 3.12 environment, ~1 GB)"; Types: full gpu8 render custom; Flags: fixed
Name: "athenea"; Description: "athenea renderer (~300 MB)"; Types: full gpu8 render custom
Name: "idol"; Description: "Idol: picture → 3D Gaussian splat (PyTorch + weights, ~8 GB; DINOv3 needs a Hugging Face token)"; Types: full gpu8 custom
Name: "eneas"; Description: "ENEAS: part masks for materials (~4 GB, + ~9 GB of weights on first use)"; Types: full gpu8 custom
Name: "iris"; Description: "Iris-3B Turbo locally: sentence → picture (~14 GB)"; Types: full custom

[Files]
Source: "{#Staging}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion; Excludes: "__pycache__\*,*.pyc"

[Dirs]
Name: "{app}\work"

[Icons]
Name: "{group}\splat2mesh"; Filename: "{app}\installer\splat2mesh.bat"; WorkingDir: "{app}"; IconFilename: "{app}\installer\splat2mesh.ico"
Name: "{group}\Install or repair components"; Filename: "{app}\installer\install_all.bat"; WorkingDir: "{app}"
Name: "{group}\Results folder"; Filename: "{app}\work"
Name: "{group}\Uninstall splat2mesh"; Filename: "{uninstallexe}"
Name: "{userdesktop}\splat2mesh"; Filename: "{app}\installer\splat2mesh.bat"; WorkingDir: "{app}"; IconFilename: "{app}\installer\splat2mesh.ico"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Desktop shortcut"

[Run]
Filename: "{app}\installer\splat2mesh.bat"; Description: "Open splat2mesh"; Flags: postinstall nowait skipifsilent shellexec

[UninstallDelete]
Type: filesandordirs; Name: "{app}\.venv"
Type: filesandordirs; Name: "{app}\tools"
; {app}\work (your results) is kept

[Code]
var
  VaePage, IrisPage: TInputOptionWizardPage;
  KeysPage: TInputQueryWizardPage;
  RunPage: TOutputProgressWizardPage;
  GpuName: String;
  GpuMB: Integer;
  Failed: String;
  CurComp: String;
  CompLo, CompHi: Integer;

procedure DetectGpu;
var
  R: TExecOutput;
  Line: String;
  P: Integer;
begin
  GpuName := ''; GpuMB := 0;
  if ExecAndCaptureOutput('nvidia-smi', '--query-gpu=name,memory.total --format=csv,noheader,nounits', '',
                          SW_HIDE, ewWaitUntilTerminated, P, R) and (GetArrayLength(R.StdOut) > 0) then
  begin
    Line := R.StdOut[0];
    P := Pos(',', Line);
    if P > 0 then
    begin
      GpuName := Trim(Copy(Line, 1, P - 1));
      GpuMB := StrToIntDef(Trim(Copy(Line, P + 1, 20)), 0);
    end;
  end;
end;

procedure InitializeWizard;
var
  Info: String;
begin
  DetectGpu;
  if GpuName = '' then
    Info := 'No NVIDIA GPU was found. Idol and ENEAS need one (8 GB or more); the renderer alone works without.'
  else
    Info := 'GPU: ' + GpuName + ', ' + IntToStr(GpuMB div 1024) + ' GB.';

  VaePage := CreateInputOptionPage(wpSelectComponents, 'Idol: image encoder',
    'Which FLUX.2 VAE Idol uses',
    'Idol encodes the picture with DINOv3 and the FLUX.2 VAE. The VAE in FLUX.2-klein-4B (Apache-2.0) has the same '
    + 'weights in bf16 as the one in FLUX.2-dev (non-commercial licence); see docs\licenses.md. DINOv3 is under '
    + 'Meta''s DINOv3 licence and is gated: accept its terms on Hugging Face before installing.', True, False);
  VaePage.Add('FLUX.2-klein-4B (Apache-2.0) — recommended, also for commercial use');
  VaePage.Add('FLUX.2-dev (non-commercial licence)');
  VaePage.SelectedValueIndex := 0;

  IrisPage := CreateInputOptionPage(VaePage.ID, 'Iris-3B: sentence → picture',
    'Where the text-to-image model runs', Info + #13#10 + #13#10
    + 'Iris-3B Turbo needs about 14 GB on the GPU with its text encoder. On smaller GPUs the text encoder (Qwen3-VL-4B) '
    + 'runs on the CPU: the same picture, a slower prompt.', True, False);
  IrisPage.Add('Local, text encoder on the CPU (8–15 GB GPUs)');
  IrisPage.Add('Local, everything on the GPU (16 GB or more)');
  IrisPage.Add('Online: the public Iris-3B Turbo Space (nothing to download; needs internet)');
  if GpuMB >= 15000 then IrisPage.SelectedValueIndex := 1
  else if GpuMB >= 7500 then IrisPage.SelectedValueIndex := 0
  else IrisPage.SelectedValueIndex := 2;

  KeysPage := CreateInputQueryPage(IrisPage.ID, 'Keys', 'Hugging Face and DeepSeek',
    'Hugging Face token: for the gated DINOv3 download only (huggingface.co/settings/tokens, after accepting '
    + 'huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m). It is used during the installation and not stored.'
    + #13#10 + 'DeepSeek API key: the part lists and checks of the materials (platform.deepseek.com). It is stored '
    + 'in your user folder (%APPDATA%\splat2mesh). Both can be left empty and given later in the app.');
  KeysPage.Add('Hugging Face token:', True);
  KeysPage.Add('DeepSeek API key:', True);
end;

function ShouldSkipPage(PageID: Integer): Boolean;
begin
  Result := False;
  if PageID = VaePage.ID then Result := not WizardIsComponentSelected('idol');
  if PageID = IrisPage.ID then Result := not WizardIsComponentSelected('iris') and not WizardIsComponentSelected('idol');
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if (CurPageID = wpSelectComponents) and (GpuName = '') and
     (WizardIsComponentSelected('idol') or WizardIsComponentSelected('eneas') or WizardIsComponentSelected('iris')) then
    Result := MsgBox('No NVIDIA GPU was found. Idol, ENEAS and Iris-3B need one. Continue anyway?',
                     mbConfirmation, MB_YESNO) = IDYES;
end;

function IrisMode: String;
begin
  case IrisPage.SelectedValueIndex of
    0: Result := 'cpu-encoder';
    1: Result := 'gpu';
  else
    Result := 'space';
  end;
  if not WizardIsComponentSelected('iris') then Result := 'space';
end;

function Vae: String;
begin
  if VaePage.SelectedValueIndex = 1 then Result := 'dev' else Result := 'klein';
end;

function UpdateReadyMemo(Space, NewLine, MemoUserInfoInfo, MemoDirInfo, MemoTypeInfo, MemoComponentsInfo,
  MemoGroupInfo, MemoTasksInfo: String): String;
begin
  Result := MemoDirInfo + NewLine + NewLine + MemoComponentsInfo + NewLine + NewLine
    + 'Options:' + NewLine + Space + 'Idol VAE: ' + Vae + NewLine + Space + 'Iris-3B: ' + IrisMode + NewLine
    + Space + 'GPU: ' + GpuName + NewLine + NewLine
    + 'The components download several GB; this takes from minutes to about an hour.';
end;

{ every output line of a step: PROGRESS lines move the bar inside the step's share }
procedure OnLog(const S: String; const Error, FirstLine: Boolean);
var
  T: String;
  P, Pct: Integer;
begin
  T := Trim(S);
  if Copy(T, 1, 9) = 'PROGRESS ' then
  begin
    T := Copy(T, 10, Length(T));
    P := Pos(' ', T);
    Pct := StrToIntDef(Copy(T, 1, P - 1), 0);
    RunPage.SetText(CurComp, Copy(T, P + 1, 200));
    RunPage.SetProgress(CompLo + (CompHi - CompLo) * Pct div 100, 1000);
  end
  else if Copy(T, 1, 7) = 'FAILED ' then
    Failed := Failed + T + #13#10;
end;

function RunStep(const Title, Exe, Params: String; Lo, Hi: Integer): Boolean;
var
  Code: Integer;
begin
  CurComp := Title; CompLo := Lo; CompHi := Hi;
  RunPage.SetText(Title, '');
  RunPage.SetProgress(Lo, 1000);
  Result := ExecAndLogOutput(Exe, Params, ExpandConstant('{app}'), SW_HIDE, ewWaitUntilTerminated, Code, @OnLog)
            and (Code = 0);
  if not Result then Failed := Failed + Title + ' (exit code ' + IntToStr(Code) + ')' + #13#10;
  RunPage.SetProgress(Hi, 1000);
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  Py, Tok, Comps: String;
  N, I, Lo, Span: Integer;
  List: array of String;
begin
  if CurStep <> ssPostInstall then Exit;
  Failed := '';
  RunPage := CreateOutputProgressPage('Installing the components', 'Each one installs under ' + ExpandConstant('{app}\tools') + '.');
  RunPage.Show;
  try
    if not RunStep('Core: uv, Python 3.12, splat2mesh', 'powershell.exe',
         '-NoProfile -ExecutionPolicy Bypass -File "' + ExpandConstant('{app}\installer\bootstrap.ps1') + '" -Root "'
         + ExpandConstant('{app}') + '"', 0, 120) then
    begin
      MsgBox('The core step failed:' + #13#10 + Failed + #13#10 + 'See the log (Setup''s /LOG option) and run '
        + '"Install or repair components" from the Start menu to retry.', mbError, MB_OK);
      Exit;
    end;
    Py := ExpandConstant('{app}\.venv\Scripts\python.exe');
    Tok := Trim(KeysPage.Values[1]);
    if Tok <> '' then
    begin
      SaveStringToFile(ExpandConstant('{tmp}\ds.txt'), Tok, False);
      RunStep('Saving the DeepSeek key', Py, '-m splat2mesh.winsetup options --deepseek-key-file "'
        + ExpandConstant('{tmp}\ds.txt') + '" --iris-mode ' + IrisMode, 120, 130);
    end
    else
      RunStep('Options', Py, '-m splat2mesh.winsetup options --iris-mode ' + IrisMode, 120, 130);

    N := 0; SetArrayLength(List, 4);
    if WizardIsComponentSelected('athenea') then begin List[N] := 'athenea'; N := N + 1; end;
    if WizardIsComponentSelected('idol') then begin List[N] := 'idol'; N := N + 1; end;
    if WizardIsComponentSelected('eneas') then begin List[N] := 'eneas'; N := N + 1; end;
    if WizardIsComponentSelected('iris') then begin List[N] := 'iris'; N := N + 1; end;
    if N > 0 then Span := 870 div N else Span := 0;
    for I := 0 to N - 1 do
    begin
      Lo := 130 + I * Span;
      Comps := '-u -m splat2mesh.winsetup install ' + List[I] + ' --vae ' + Vae + ' --iris-mode ' + IrisMode;
      Tok := Trim(KeysPage.Values[0]);
      if (List[I] = 'idol') and (Tok <> '') then
      begin
        SaveStringToFile(ExpandConstant('{tmp}\hf.txt'), Tok, False);       { read and deleted by winsetup }
        Comps := Comps + ' --hf-token-file "' + ExpandConstant('{tmp}\hf.txt') + '"';
      end;
      RunStep(List[I], Py, Comps, Lo, Lo + Span);
    end;
    DeleteFile(ExpandConstant('{tmp}\hf.txt')); DeleteFile(ExpandConstant('{tmp}\ds.txt'));
    if Failed <> '' then
      MsgBox('Some steps did not finish:' + #13#10 + Failed + #13#10
        + 'The rest is installed. Retry from the app (Components) or "Install or repair components" in the Start menu.',
        mbInformation, MB_OK);
  finally
    RunPage.Hide;
  end;
end;
