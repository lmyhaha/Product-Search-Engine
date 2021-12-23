#coding=UTF-8
import web
import SearchFiles as se

urls = (
    '/', 'index',
    '/s', 'text',
    '/i', 'img'
)

render = web.template.render('templates')

class index:
    def GET(self):
        return render.formtest()

class text:
    def GET(self):
        user_data = web.input()
        print user_data
        x = user_data.name
        seq, term = se.func(x)
        return render.result("web", x, seq, term)

class img:
    def POST(self):
        x = web.input(myfile={})
        if 'myfile' in x:
            filepath = x.myfile.filename
            print filepath
            filename = filepath.split('/')[-1]
            fout = open('static/rubbish/'+filename, 'wb')
            fout.write(x.myfile.file.read())
            fout.close()
        seq, term = se.func2('static/rubbish/'+filename)
        return render.result("img", filepath, seq, term)

if __name__ == "__main__":
    app = web.application(urls, globals())
    app.run()
