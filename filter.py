#filtered images - 1977, Amaro, Apollo, Brannan, Earlybird, Gotham, Hefe, Hudson, Inkwell, Lofi, LordKelvin, Mayfair, Nashville, Poprocket, Rise, Sierra, Sutro, Toaster, Valencia, Walden, Willow, and XProII
from PIL import Image
import pilgram  #support:_1977, brannan, earlybird, hudson, inkwell, kelvin, lofi, mayfair, nashville, rise, toaster, valencia, walden, willow, xpro2
import os,sys
import numpy as np 

#Amaron
def get_coeff( points ):

    x = []
    y = []
    R = []

    for p in points:
        x.append( p[0] )
        y.append( p[1] ) 

    l = len(points)-1
    for i in range(0, l+1):
        print(i)
        if i == 0:
            p = 1
        elif i == l:
            p = l-1
        else:
            p = i
        xl = (x[p-1], x[p], x[p+1])
        yl = (y[p-1], y[p], y[p+1])
        
        R.append( np.polyfit( xl, yl, 2) )

    return R


def get_right_coeff( points, xval ):

    pos = 0
    nr = 0
    for i in points:
        if xval >= i[0]:
            pos = nr
        nr += 1

    return pos

def get_new_val( coeff, x ):


    Y =  coeff[0]*x**2 + coeff[1]*x + coeff[2]

    return Y


def filters_points( filter_name ):

    points = {
        'amaro' : {
            'r' : [[0,19],[30,62],[82,148],[128,188],[145,200],[255,250]],
            'g' : [[0,0],[48,72],[115,188],[160,220],[233,245],[255,255]],
            'b' : [[0,25],[35,80],[106,175],[151,188],[215,215],[240,235],[255,245]]
        },
    }

    return points.get( filter_name )


def get_image( pic_url ):
    
    jpgfile = Image.open( pic_url )
    data = jpgfile.load()

    width = jpgfile.size[0]
    height = jpgfile.size[1]
    
    image = []
    image.append( width )
    image.append( height )
    image.append( list(jpgfile.getdata()) )

    return image;


def replace_image_colors( pixels, filter_name ):

    newdata = []

    filter_points = filters_points( filter_name )
    coeff_r = get_coeff( filter_points.get('r') )
    coeff_g = get_coeff( filter_points.get('g') )
    coeff_b = get_coeff( filter_points.get('b') )
    print(coeff_r)
    for r,g,b in pixels:

        a = (r + g + b)/3
        #print (get_right_coeff(filter_points.get('r'), r))
        r = int( get_new_val(coeff_r[ get_right_coeff(filter_points.get('r'), r) ], r) )
        g = int( get_new_val(coeff_g[ get_right_coeff(filter_points.get('g'), g) ], g) )
        b = int( get_new_val(coeff_b[ get_right_coeff(filter_points.get('b'), b) ], b) )
       
        new_colors = (r,g,b)

        newdata.append(new_colors)

    return newdata


def save_image( new_pic_url, pixels, width, height ):

    newImage = Image.new("RGB", (width, height), "white")
    newImage.putdata( pixels )
    newImage.save( new_pic_url + ".jpg" )
    return newImage


def apply_filter( pic_url , filter_name ):

    image = get_image( pic_url )
    new_pixels = replace_image_colors( image[2], filter_name ) 
    newImage = save_image( pic_url[:-4] + '-' + filter_name, new_pixels, image[0], image[1] )
    return newImage
#apply_filter( 'bt21.jpg', 'amaro' ) 

def rotate_image( pic_url, rot ):

    file = Image.open( pic_url )
    out = file.rotate(rot, expand = 1)
    out.save( pic_url )

def test_curve(filter_name, color):

    newdata = []

    filter_points = filters_points( filter_name )

    for x in range(0, 255):
        for y in range(0, 255):
    
       
            coeff = get_coeff( filter_points.get(color), x )
            val = int( get_new_val(coeff, x) )

            if y == val:
                new_colors = (0,0,0)
            else:
                new_colors = (255,255,255)

            newdata.append(new_colors)

    save_image( 'test-' + filter_name, newdata, 255, 255 )

def filter_process(filter_no):
    switcher = {
        0: pilgram._1977(im),
        1: pilgram.brannan(im),
        2: pilgram.earlybird(im),
        3: pilgram.hudson(im),
        4: pilgram.inkwell(im),
        5: pilgram.kelvin(im),
        6: pilgram.lofi(im),
        7: pilgram.mayfair(im),
        8: pilgram.nashville(im),
        9: pilgram.rise(im),
        10: pilgram.toaster(im),
        11: pilgram.valencia(im),
        12: pilgram.walden(im),
        13: pilgram.willow(im),
        14: pilgram.xpro2(im),
        15: apply_filter('bt21.jpg', 'amaro' ) 
    }
    return switcher.get(filter_no, "nothing")
im = Image.open('bt21.jpg')
new_im = filter_process(15)
new_im.show()

#new_im = pilgram._1977(im).save('bt21-_1977.jpg')
#new_im = pilgram._1977(im)
#new_im.show()

#Gotham, Apollo, Hefe, Poprocket, Sierra, Sutro