const fs = require('fs');

var gplay = require('google-play-scraper');

const writeToFile = (fileName) => (data) => {
  fs.writeFile(fileName, JSON.stringify(data), 'utf8', function(err) {
      if (err) throw err;
      console.log('complete');
    }
  );
}

gplay.reviews({
  appId: 'GOOGLE.APK.NAME',
  sort: gplay.sort.DATE,
  num: 30000
}).then(writeToFile('reviews.json'), console.error)