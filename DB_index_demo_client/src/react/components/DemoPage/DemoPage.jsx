import React from 'react';
import { hot } from 'react-hot-loader';
import './DemoPage.styl';


import Recognition from './recognition';


class App extends React.Component{
    constructor(props, context) {
        super(props, context);
    
        this.render = this.render.bind(this);
    
        this.state = {
            mounted: true,
            queryImageSelected: false,
            queryImageList: []
        }
    }
   
    componentDidMount() {
        this.serverRequest = $.get('/api/testImageList', (data, status) => {
            this.setState({ mounted: true,
                queryImageList: data.list
            });
            console.log(data)
            console.log(this.state);
        }, 'json'); 


        this.setState({ mounted: true
        });


    }

    render = () => (    
        <Recognition queryImageList={this.state.queryImageList}/>
    );
}





export default hot(module)(App);      