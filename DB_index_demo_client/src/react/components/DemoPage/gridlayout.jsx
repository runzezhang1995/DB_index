import React from 'react';
import { hot } from 'react-hot-loader';
import _ from 'lodash';

import './gridlayout.css';

import {Image} from 'react-bootstrap';
import { Responsive, WidthProvider } from "react-grid-layout";

const ResponsiveReactGridLayout = WidthProvider(Responsive);

const image_base_url = 'http://99.23.139.146:1551/image/'



class ImageGrid extends React.Component { 
    constructor(props, context) {
        super(props, context);

        this.generateDOM = this.generateDOM.bind(this);

        this.state = {
            currentBreakpoint: "lg",
            compactType: "vertical",
            mounted: false,
            layouts: { lg: this.props.initialLayout }
        };
    }
    

    static defaultProps = {
        className: "layout",
        rowHeight: 154,
        onLayoutChange: (layout) => {
            console.log('layout change');
            console.log(layout);
        },
        cols: { lg: 5, md: 5, sm: 5, xs: 5, xxs: 5 },
        initialLayout: generateLayout()
    };

    




    componentDidMount() {
        this.setState({ mounted: true });
    }
    

    generateDOM() {
    const Srcs = this.props.imageSrcs;
    return _.map(this.state.layouts.lg, function(l, i) {
        const src =Srcs[i]
        return (
        <div key={i}>
            {/* <span className="text">{i}</span> */}
            <div className={src.correct? "bg-green" : "bg-red"} >            
                <Image src={image_base_url + src.image} alt="image not found" srcSet="" className="gallery-image" fluid rounded/>
                <h5 style={{"textAlign": "center", "margin":"auto"}}>{src.score.toFixed(3)}</h5>
            </div>
        </div>
        );
    });
}

    render= () => (
        <div>
            <ResponsiveReactGridLayout
                {...this.props}
                layouts = {this.state.layouts}
                
                isDraggable ={false}
                isResizable ={false}
                measureBeforeMount = {false}
                useCSSTransforms={this.state.mounted}
                compactType={this.state.compactType}
                preventCollision={!this.state.compactType}
            >
                {this.generateDOM()}
            </ResponsiveReactGridLayout>
        </div>
    );
}




function generateLayout() {
    return _.map(_.range(0, 10), (item, i) => {
        return {
            x: i % 5,
            y: parseInt(i / 5),
            w: 1,
            h: 1,
            i: i.toString(),
            static: false
        };
    });
}


export default hot(module)(ImageGrid);      