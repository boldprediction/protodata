import numpy as np
import cortex, time
from cortex import mni


acc = np.load('uniform_bert2_combined_accuracies_subj_space.npy')

subjects = acc.item()['acc'].keys()

surface = 'fMRI_story_{}'
xfm = '{}_ars'

mni_transforms = dict()
mni_masked = dict()

subjects = ['F','G','H','I','J','K','L','M','N']

for subject in subjects:
	start = time.time()
	mni_transforms[subject] = cortex.db.get_mnixfm(surface.format(subject),
												   xfm.format(subject))
	print('transform for {}, {} seconds'.format(subject, time.time()-start))


mask_mni = np.load('mask_MNI.npy')
#cortex.db.get_mask('MNI', 'atlas','thin')
n_v_mni = mask_mni.sum()

errors_files = []


for subject in subjects:
	start = time.time()
	mask = cortex.db.get_mask(surface.format(subject), xfm.format(subject),'thick')
	nr = len(acc.item()['acc'][subject])
	mni_masked[subject] = np.zeros((nr,n_v_mni))
	for ir in range(nr):
		tmp = np.vstack(acc.item()['acc'][subject])[ir]
		vol = np.zeros(mask.shape)
		vol[mask] = tmp
		vol = cortex.Volume(tmp, surface.format(subject), xfm.format(subject),mask = mask)
		try:
			mni_vol = mni.transform_to_mni(vol,mni_transforms[subject]).get_data().T
			mni_masked[subject][ir] = mni_vol[mask_mni]
		except:
			errors_files.append('subject {} row {}'.format(subject, ir))
		if ir%10 ==1:
			print('time_left for {}, {} seconds'.format(subject, 
														(time.time()-start)/(ir+1)*(nr-ir-1)))


for subject in subjects:
	np.save('bert2_uniform_mni_{}.npy'.format(subject),mni_masked[subject])