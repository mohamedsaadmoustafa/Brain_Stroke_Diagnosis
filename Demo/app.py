import os
import cv2
import base64
from io import BytesIO

from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
from flask import Flask

from predict import *


app = Flask(__name__)
app.secret_key = "mohamed" 

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024



def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in set(['npy', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
	
@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
#@app.route('/pred')
def upload_file():
	if request.method == 'POST':
        # check if the post request has the file part
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
			
		file = request.files['file']
		
		if file.filename == '':
			flash('No file selected for uploading')
			return redirect(request.url)
			
		if file and allowed_file( file.filename ):
			filename = secure_filename(file.filename)
			file.save( os.path.join(app.config['UPLOAD_FOLDER'], filename ) )

			flash( f"File Name {filename}" )
			
			x, y = predict_f( os.path.join(app.config['UPLOAD_FOLDER'], filename ) )

			# Threshold predictions
			y_threshold  = ( y > 0.5 ).astype( np.uint8 )
			# show masked image
			kernel = np.ones( ( 5, 5 ), np.uint8 ) ############## change kernel shape
			y_threshold = cv2.dilate( y_threshold * 255., kernel, iterations = 1 )
			
			
			fig, ax = plt.subplots( 1, 4, figsize = ( 13, 3 ) )

			ax[0].imshow( x )
			ax[1].imshow( y )
			ax[2].imshow( y_threshold );
			ax[3].imshow( x );
			ax[3].imshow( np.ma.masked_where( y_threshold == 1, y_threshold ), alpha=0.5 );

			ax[0].set_title( 'MRI Slice' )
			ax[1].set_title( 'Predicted' )
			ax[2].set_title( 'Threeshold & Dilation' );
			ax[3].set_title( 'Lesion Area' );

			#plt.savefig( 'static/images/plot.png' )
			img = BytesIO()
			plt.savefig(img, format='png')
			plt.close()
			img.seek(0)
			plot_url = base64.b64encode(
				img.getvalue()
			).decode( 'utf8' )

			return render_template( 'pred.html', plot_url = plot_url ) 

			
		else:
			flash( 'Allowed file types are npy' )
			return redirect(request.url)


if __name__ == "__main__":
	app.debug = True
	app.run()