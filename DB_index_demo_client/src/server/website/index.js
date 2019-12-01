import express from 'express';
import fs from 'fs';

const router = express.Router();


router.get('/', (req, res) => {
    res.render('react', {
        js_host: global.jsHost,
        name: 'demo',
        title: 'Facial Database Index Demo',
    });
});



// router.get('/demo/:config', (req, res) => {
//     res.render('react', {
//         js_host: global.jsHost,
//         name: 'demo',
//         title: 'People Detection & Re-identification',
//     });
// });


// router.get('/login', (req, res) => {
//     res.render('react', {
//         js_host: global.jsHost,
//         name: 'login',
//         title: 'Login',
//     });
// });


// router.get('*', (req, res) => {
//     res.redirect('/');
// });



export default router;