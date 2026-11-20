// Recording-specific table-region extraction from the completed native SPZ.
// Adapt the bounds/color gates for another scene; no API credentials are used.
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const modules=process.env.ATLAS_RENDER_NODE_MODULES??'/tmp/worldlabs_viewer/node_modules';
const {chromium}=await import(pathToFileURL(path.join(modules,'playwright/index.mjs')).href);
const browser=await chromium.launch({executablePath:process.env.ATLAS_CHROME??'/usr/bin/google-chrome',headless:true,args:['--no-sandbox','--disable-gpu-sandbox','--use-gl=angle','--use-angle=gl','--ignore-gpu-blocklist']});
try{
 const page=await browser.newPage();
 await page.goto((process.env.ATLAS_VIEWER_URL??'http://127.0.0.1:8870/')+'?inspect=1');
 await page.waitForFunction(()=>window.viewerReady===true,null,{timeout:120000});
 const options={min:[-1.5,.62,-1.4],max:[.6,.88,1.4],minOpacity:.3,maxColorRange:.25,minBrightness:.55};
 const rows=await page.evaluate(o=>window.inspectSplats(o),options);
 const output='outputs/worldlabs/table_splats.json';
 fs.writeFileSync(output,JSON.stringify(rows));
 fs.writeFileSync('outputs/worldlabs/table_splat_extraction.json',JSON.stringify({options,count:rows.length,coordinateSystem:'Atlas native RUB / Y-up',source:'Completed refined Atlas SPZ; appearance-derived table region'},null,2));
 console.log('Extracted',rows.length,'table-region splats');
}finally{await browser.close();}
