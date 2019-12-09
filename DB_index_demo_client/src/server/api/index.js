import express from 'express';
import fs from 'fs';
import request from 'request';
import http from 'http';

const router = express.Router();

const base_url = 'http://127.0.0.1:1551';

router.get('/testImageList', (req, res) => {
    const url = base_url + '/list';
    
    const handler = (body) => {
        const body_obj = JSON.parse(body)
    
        res.send({
            success:true,
            list: body_obj.image_list
        })
    };
    
    request(url, function (error, response, body) {
        if (error) throw new Error(error);
        handler(body)
    });
    
});



router.post('/recognize/:imageName', (req, res) => {
    console.log('recognize for image')
    const imageName = req.params.imageName
    const url =  base_url + '/recognize/' + imageName;
    const handler = (body) => {
        const body_obj = JSON.parse(body)
        res.send(body_obj)
    };

    request.post({
        headers: {'content-type' : 'application/x-www-form-urlencoded'},
        url
      }, function(error, response, body){
        if (error) throw new Error(error);
        handler(body)
      });
})


export default router;