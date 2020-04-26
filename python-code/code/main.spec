# -*- mode: python -*-
options = [ ('v', None, 'OPTION')]
block_cipher = None


a = Analysis(['main.py'],
             pathex=['D:\\Projects\\Face\\Python\\nssm\\py36-venv','D:\\Projects\\Face\\Python\\nssm\\py36-venv\\Lib\\site-packages\\cv2'],
             binaries=[],
             datas=[],
             hiddenimports=['numpy.random.common', 'numpy.random.bounded_integers', 'numpy.random.entropy','numpy.core._dtype_ctypes'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='main',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=True )
